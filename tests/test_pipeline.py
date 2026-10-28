"""Per-persona pipeline: the measured `mentioned` must never be decided by the
sentiment model's say-so (invariant 2)."""
import pytest

from app.models import Competitor
from app.services import pipeline
from app.services.pipeline import verified_alias

ANSWER = "For a family cruise from Singapore, Royal Caribbean is the usual pick. The Disney Wish is pricier."
RC = Competitor(name="Royal Caribbean", match_names=["Royal Caribbean"])


@pytest.mark.parametrize(
    "candidate, expected",
    [
        ("cruise", None),  # generic word, never written like a name
        ("Disney Wish", "Disney Wish"),  # a real product name in the answer
        ("Disney Treasure", None),  # not in the answer
        ("Royal Caribbean", None),  # another entity's name
        ("Ro", None),  # too short
        (None, None),
    ],
)
def test_verified_alias(candidate, expected):
    assert verified_alias(candidate, ANSWER, ["Royal Caribbean"]) == expected


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
