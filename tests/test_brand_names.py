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
