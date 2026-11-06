"""Which names count as the brand: product names always, ambiguous parent
names ("Disney") only in a sentence about the category, and model-suggested
model codes ("A20") only when the sentence also names the brand."""
import pytest

from app.services.pipeline import verified_alias
from app.services.scoring import BrandContext, count_brand_mentions, find_mentions

# Text of travel.yahoo.com/cruises/article/best-family-friendly-cruises (live,
# trimmed): reported reading 0 Disney mentions though a browser finds them.
YAHOO = (
    "Trending: Disney reveals Monstropolis coaster. "
    "Disney is an obvious choice, but don't overlook Royal Caribbean's mega ships with entertainment for all ages. "
    "Disney Wish, Port Canaveral, Fla. If your little Mouseketeers are ready to hit the high seas, Disney Wish is a "
    "fabulous choice. Cruising truly doesn't get more family-friendly than a Disney ship."
)
DCL = ["Disney Cruise Line", "Disney Cruise", "DCL"]
CRUISE = BrandContext.of(["Disney"], ["cruise", "ship", "sail", "cabin"])


def test_the_yahoo_page_counts_product_names_and_disney_in_cruise_context():
    assert count_brand_mentions(YAHOO, "Disney Cruise Line", DCL) == 0  # the reported bug
    assert count_brand_mentions(YAHOO, "Disney Cruise Line", [*DCL, "Disney Wish"]) == 2
    assert count_brand_mentions(YAHOO, "Disney Cruise Line", [*DCL, "Disney Wish"], context=CRUISE) == 4


def test_a_context_name_never_counts_outside_its_category():
    assert count_brand_mentions("Disney reveals Monstropolis coaster.", "Disney Cruise Line", DCL, context=CRUISE) == 0
    assert count_brand_mentions("We watched Disney+ on the plane.", "Disney Cruise Line", DCL, context=CRUISE) == 0
    # and not at all without category words to check against
    assert BrandContext.of(["Disney"], []) is None


def test_a_full_name_wins_over_the_context_name_at_the_same_place():
    mentions = find_mentions("Disney Wish sails from Florida.", "Disney Cruise Line", [], ["Disney Wish"], context=CRUISE)
    assert [(m.start, m.end) for m in mentions] == [(0, 11)]


def test_category_words_match_their_inflections():
    assert count_brand_mentions("Disney cruises are pricey.", "Disney Cruise Line", ["DCL"], context=CRUISE) == 1
    assert count_brand_mentions("We loved sailing with Disney.", "Disney Cruise Line", ["DCL"], context=CRUISE) == 1


@pytest.mark.parametrize(
    "candidate, answer, own, expected",
    [
        # model code named alongside the brand: counted again (regressed by the ownership check)
        ("A20", "For under $200, Samsung's A20 is solid. The Redmi Note 13 is cheaper.", ["Samsung"], "A20"),
        ("Galaxy A20", "Samsung's Galaxy A20 is solid.", ["Samsung"], "Galaxy A20"),
        # no corroboration in the sentence: left to the detect-time names
        ("A20", "For under $200, the A20 is solid. Samsung also makes tablets.", ["Samsung"], None),
        # a rival's unlisted product is still never taken on the model's word
        ("Icon of the Seas", "Royal Caribbean's Icon of the Seas suits families. Disney Cruise Line is pricier.",
         ["Disney Cruise Line"], None),
    ],
)
def test_model_suggested_names(candidate, answer, own, expected):
    assert verified_alias(candidate, answer, own, ["Royal Caribbean", "Xiaomi", "Redmi"]) == expected


def test_detect_returns_context_names_and_clean_category_terms():
    from app.services.detect import _category_terms, _fit_names, _as_names

    assert _fit_names(_as_names(["Disney", " ", "D"])) == ["Disney"]
    assert _category_terms(["Cruise", "ship", "cruise", "go", "two words", "sail"]) == ["cruise", "ship", "sail"]


def test_prompt_screen_treats_context_names_as_a_leak():
    from app.models import PromptsRequest
    from app.routes.prompts import _screen_names
    from app.services.prompt_gen import names_a_brand

    body = PromptsRequest(brand="Disney Cruise Line", industry="Cruises", accessCode="A",
                          personas=[{"id": "p1", "title": "T", "role": "r", "pains": "p", "criteria": "c"}],
                          brandMatchNames=["Disney Cruise Line"], brandContextNames=["Disney"], categoryTerms=["cruise"])
    assert names_a_brand("Is Disney worth it for toddlers?", body.brand, [], _screen_names(body))


# ── names that are also ordinary words ───────────────────────────────

from app.models import Competitor as _Competitor
from app.services.scoring import find_mentions as _find


ON = _Competitor(name="On Running", match_names=["On Running", "On", "On Cloud"])


def _rivals(text, caseless=False):
    return [text[m.start:m.end] for m in _find(text, "Nike", [ON], ["Nike"], caseless=caseless) if m.kind == "competitor"]


def test_a_function_word_name_is_not_the_word_in_passing():
    # Confirmed live: "based on performance" counted as a mention of On Running.
    assert _rivals("Pick shoes based on cushioning and fit.") == []
    assert _rivals("On long runs, cushioning matters more than weight.") == []  # opens a sentence
    assert _rivals("For long runs, try Nike or On, both cushioned.") == ["On"]
    assert _rivals("The On Cloud is light, and On Running makes it.") == ["On Cloud", "On Running"]


def test_a_capitalised_one_word_name_needs_the_capital_in_prose():
    grab = ["Grab"]
    assert _find("Grab is the default here.", "Grab", [], grab)  # opening a sentence still counts
    assert _find("Riders mostly use Grab here.", "Grab", [], grab)
    assert not _find("Riders grab whatever app is cheapest.", "Grab", [], grab)
    # A name that isn't an English word counts lowercase too (confirmed live: "adidas").
    assert _find("Try the adidas Adizero or a Hoka.", "Adidas", [], ["Adidas"])


def test_text_without_casing_matches_in_any_case_but_never_function_words():
    # URL slugs and searches typed in lowercase carry no casing.
    assert _find("xero vs quickbooks review", "Xero", [], ["Xero"])
    assert _rivals("best shoes on sale") == []
    # Leak screening is caseless on purpose, even in mixed-case text.
    assert _find("Best app like notion for my Cafe?", "Notion", [], ["Notion"], caseless=True)
    assert not _find("Best app like notion for my Cafe?", "Notion", [], ["Notion"])  # the word, in prose



# ── text that must not name the brand ────────────────────────────────

from app.services.scoring import without_names


def test_buyer_context_loses_brand_parent_and_rival_names():
    rc = _Competitor(name="Royal Caribbean", match_names=["Royal Caribbean"])
    names = ["Disney Cruise Line", "Disney"]
    # Confirmed live for Disney Cruise Line.
    assert without_names("Families, couples, and Disney fans choosing a cruise vacation.", "Disney Cruise Line",
                         [rc], names) == "Families, couples, and fans choosing a cruise vacation."
    assert without_names("Disney's fans and Royal Caribbean regulars comparing ships", "Disney Cruise Line",
                         [rc], names) == "fans and regulars comparing ships"
    plain = "Families choosing a cruise holiday"
    assert without_names(plain, "Disney Cruise Line", [rc], names) is plain
    # "on" in passing is not On Running (the function-word rule applies here too).
    assert without_names("Runners choosing shoes based on fit", "Nike", [ON], ["Nike"]) == (
        "Runners choosing shoes based on fit")



# ── brands outside the competitor list ───────────────────────────────

from app.models import Sentiment as _Sentiment
from app.services.scoring import analyze_answer as _analyze


def test_rank_counts_unlisted_brands_found_in_the_text():
    # Confirmed live: "Saucony, Brooks, Nike" ranked Nike #1.
    text = "Try the Saucony Endorphin or Brooks Ghost; the Nike Pegasus is a solid third pick, ahead of Hoka."
    hoka = _Competitor(name="Hoka", match_names=["Hoka"])
    mentioned, rank, _, parts, names = _analyze(
        text=text, brand="Nike", competitors=[hoka], sentiment=_Sentiment.positive, brand_match_names=["Nike", "Pegasus"],
        other_names=["Saucony", "Brooks", "Nike", "Hoka", "Runners Point", "Puma"],
    )
    assert mentioned and rank == 3
    assert names == ["Saucony", "Brooks", "Nike", "Hoka"]  # Runners Point and Puma aren't in the text
    assert {p.name for p in parts if p.name} == {"Nike", "Hoka"}  # only known names are highlighted


def test_an_unlisted_name_that_is_a_known_name_variant_is_not_a_new_brand():
    text = "QuickBooks Online is popular; Xero is simpler."
    qb = _Competitor(name="QuickBooks", match_names=["QuickBooks"])
    _, rank, _, _, names = _analyze(text=text, brand="Xero", competitors=[qb], sentiment=_Sentiment.neutral,
                                     brand_match_names=["Xero"], other_names=["QuickBooks Online", "Xero"])
    assert names == ["QuickBooks", "Xero"] and rank == 2
