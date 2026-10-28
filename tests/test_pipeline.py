"""Per-persona pipeline: the measured `mentioned` must never be decided by the
sentiment model's say-so (invariant 2)."""
import pytest

from app.models import Competitor
from app.services import pipeline
from app.services.pipeline import verified_alias

ANSWER = "For a family cruise from Singapore, Royal Caribbean is the usual pick. The Disney Wish is pricier."
RC = Competitor(name="Royal Caribbean", match_names=["Royal Caribbean"])


@pytest.mark.parametrize(
    "candidate, answer, expected",
    [
        ("cruise", ANSWER, None),  # generic word, never written like a name
        ("Disney Wish", ANSWER, "Disney Wish"),  # a real product name sharing the brand's head word
        ("Disney Treasure", ANSWER, None),  # not in the answer
        ("Royal Caribbean", ANSWER, None),  # another entity's name
        ("Ro", ANSWER, None),  # too short
        (None, ANSWER, None),
        # a rival's unlisted ship: in the answer, written like a name, but not the brand's
        ("Icon of the Seas", "Royal Caribbean's Icon of the Seas suits families.", None),
        # capitalised only because it opens a sentence / a bullet
        ("Cruise", "Cruise lines from Singapore: Royal Caribbean.", None),
        ("Cruise", "Options:\n- Cruise packages with Royal Caribbean", None),
        ("Disney", "Many pick Disney for kids.", "Disney"),
    ],
)
def test_verified_alias(candidate, answer, expected):
    assert verified_alias(candidate, answer, ["Disney Cruise Line"], ["Royal Caribbean"]) == expected


def test_overlap_check_is_whole_word():
    # 'Dis' inside 'Disney' is not an overlap with a rival called 'Dis'.
    assert verified_alias("Disney Wish", ANSWER, ["Disney Cruise Line"], ["Dis"]) == "Disney Wish"


def _fake_calls(monkeypatch, answer: str, sentiment: dict):
    async def fake_text(**_):
        return answer

    async def fake_structured(**_):
        return sentiment

    monkeypatch.setattr(pipeline, "call_text", fake_text)
    monkeypatch.setattr(pipeline, "call_structured", fake_structured)


async def _analyze():
    return await pipeline._analyze_one_prompt(
        prompt="q", brand="Disney Cruise Line", brand_match_names=["Disney Cruise Line"], competitors=[RC],
        buyer_context=None, market=None, api_key="k", answer_model="a", fast_model="f",
    )


async def test_generic_mention_text_does_not_make_the_brand_mentioned(monkeypatch):
    answer = "For a family cruise from Singapore, Royal Caribbean is the usual pick."
    _fake_calls(monkeypatch, answer, {"sentiment": "Positive", "quote": answer, "mentionText": "cruise",
                                      "competitorMentions": []})
    result = await _analyze()
    assert (result.mentioned, result.rank, result.vis, result.quote) == (False, None, 0, "")


async def test_real_product_name_counts_as_a_mention(monkeypatch):
    _fake_calls(monkeypatch, ANSWER, {"sentiment": "Neutral", "quote": "", "mentionText": "Disney Wish",
                                      "competitorMentions": []})
    result = await _analyze()
    assert result.mentioned and result.rank == 2


async def test_competitor_mention_text_cannot_overlap_the_brand_name(monkeypatch):
    _fake_calls(monkeypatch, ANSWER, {"sentiment": "Neutral", "quote": "", "mentionText": "",
                                      "competitorMentions": [{"competitor": "Royal Caribbean",
                                                              "mentionText": "Disney"}]})
    result = await _analyze()
    assert [p.text for p in result.parts if p.kind == "competitor"] == ["Royal Caribbean"]


async def test_one_failed_prompt_keeps_the_personas_other_answers(monkeypatch):
    from app.models import PersonaIn

    async def flaky_text(*, user, **_):
        if user == "bad":
            raise RuntimeError("upstream 500")
        return ANSWER

    async def fake_structured(**_):
        return {"sentiment": "Neutral", "quote": "", "mentionText": "", "competitorMentions": []}

    monkeypatch.setattr(pipeline, "call_text", flaky_text)
    monkeypatch.setattr(pipeline, "call_structured", fake_structured)
    persona = PersonaIn(id="p1", title="T", role="r", pains="p", criteria="c")
    kwargs = dict(persona=persona, brand="Disney Cruise Line", brand_match_names=["Disney Cruise Line"],
                  competitors=[RC], buyer_context=None, market=None, api_key="k")

    result = await pipeline.run_persona(prompts=["good", "bad", "good too"], **kwargs)
    assert [ex.prompt for ex in result.exchanges] == ["good", "good too"]
    assert result.failed_prompts == 1

    with pytest.raises(RuntimeError):
        await pipeline.run_persona(prompts=["bad"], **kwargs)


def test_persona_sentiment_ignores_prompts_that_never_named_the_brand():
    from app.models import Sentiment
    from app.services.pipeline import _aggregate, _PromptAnalysis

    def analysis(mentioned, sentiment):
        return _PromptAnalysis(prompt="q", mentioned=mentioned, sentiment=sentiment, rank=1 if mentioned else None,
                               vis=50 if mentioned else 0, quote="", parts=[])

    result = _aggregate([analysis(True, Sentiment.positive), analysis(False, Sentiment.neutral),
                         analysis(False, Sentiment.neutral)])
    assert result.sentiment == Sentiment.positive
