import csv
import json

import pytest

from app.services import grounding, inventory
from app.services import source_quality as sq

HEADER = ["Country", "App/Site name", "Creative Type", "Bid requests", "Estimated CPM"]


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)


# ── parsing the DSP export ───────────────────────────────────────────

def test_parse_merges_creative_types_skips_apps_and_reports_unresolved(tmp_path):
    p = tmp_path / "x.csv"
    _write_csv(p, [
        ["Singapore", "AsiaOne, Asia's Leading News Portal (asiaone.com)", "Banner", "30000000", "0.42 - 0.84"],
        ["Singapore", "AsiaOne, Asia's Leading News Portal (asiaone.com)", "Video", "500000", "1.10 - 2.20"],
        ["Singapore", "sethlui.com", "Banner", "4000000", "0.16 - 0.32"],
        ["Singapore", "Candy Crush Saga (com.king.candycrushsaga)", "Banner", "3000000", "1.00 - 2.00"],
        ["Singapore", "TuneIn Radio (tunein.player)", "Banner", "3000000", "1.00 - 2.00"],
        ["Singapore", "Yahoo Mail (577586159)", "Banner", "400000", "1.56 - 3.12"],
        ["Singapore", "merriam webster.com", "Banner", "900000", "0.40 - 0.80"],
        ["Singapore", "Broken Row (x.com)", "Banner", "not-a-number", "0.1 - 0.2"],
    ])
    r = inventory.parse_csv(p)
    assert [s["domain"] for s in r.sites] == ["asiaone.com", "sethlui.com"]  # ordered by volume
    asiaone = r.sites[0]
    assert set(asiaone["creative"]) == {"Banner", "Video"}
    assert asiaone["bidRequests"] == 30_500_000
    assert asiaone["creative"]["Video"] == {"bidRequests": 500000, "cpmMin": 1.1, "cpmMax": 2.2}
    assert asiaone["name"] == "AsiaOne, Asia's Leading News Portal"
    assert {a["id"] for a in r.apps} == {"com.king.candycrushsaga", "tunein.player", "577586159"}
    assert {a["id"]: a["platform"] for a in r.apps}["577586159"] == "ios"
    assert [u["name"] for u in r.unresolved] == ["merriam webster.com"]  # kept whole, not guessed at
    assert r.bad_rows == 1


def test_app_rows_merge_per_app_across_creative_types(tmp_path):
    p = tmp_path / "x.csv"
    _write_csv(p, [
        ["Singapore", "TuneIn Radio (tunein.player)", "Banner", "6000000", "0.68 - 1.36"],
        ["Singapore", "TuneIn Radio (tunein.player)", "Video", "3000000", "1.72 - 3.44"],
        ["Singapore", "TuneIn Radio (418987775)", "Banner", "1000000", "0.60 - 1.20"],
    ])
    r = inventory.parse_csv(p)
    by_id = {a["id"]: a for a in r.apps}
    assert set(by_id["tunein.player"]["creative"]) == {"Banner", "Video"}
    assert by_id["tunein.player"]["bidRequests"] == 9_000_000
    assert by_id["418987775"]["platform"] == "ios"


def test_repair_candidates_and_only_verified_repairs_are_added(tmp_path):
    assert inventory.repair_candidates("merriam webster.com") == ["merriam-webster.com", "merriamwebster.com"]
    p = tmp_path / "x.csv"
    _write_csv(p, [
        ["Singapore", "merriam webster.com", "Banner", "900000", "0.40 - 0.80"],
        ["Singapore", "aov news.com", "Banner", "800000", "0.30 - 0.60"],
    ])
    r = inventory.parse_csv(p)
    still = inventory.add_repaired(r, {"merriam webster.com": "merriam-webster.com"})  # only this one verified
    assert [s["domain"] for s in r.sites] == ["merriam-webster.com"]
    assert r.sites[0]["creative"]["Banner"]["bidRequests"] == 900000
    assert [u["name"] for u in still] == ["aov news.com"]


def test_sitelist_fields_include_in_app_reach(monkeypatch):
    site = {"domain": "myfitnesspal.com", "name": "MyFitnessPal", "bidRequests": 0, "creative": {},
            "app": {"bidRequests": 2_780_095, "creative": {"Banner": {"bidRequests": 2_780_095, "cpmMin": 0.4, "cpmMax": 1.8}}}}
    monkeypatch.setattr(inventory, "load_sites", lambda: {"asiaone.com": SITES["asiaone.com"], "myfitnesspal.com": site})
    app_only = inventory.sitelist_fields("myfitnesspal.com")
    assert app_only["inventory"] == ["In-app Banner"] and app_only["availability"] == "2,780,095 in-app bid requests"
    assert app_only["cpm"] == "0.40 - 1.80" and app_only["buyable"]
    assert inventory.sitelist_fields("asiaone.com")["inventory"] == ["Banner", "Video"]  # no app data on this one


@pytest.mark.parametrize("token", ["com.livescore", "tunein.player", "bbc.mobile.news.ww", "com.king.candycrushsaga"])
def test_android_packages_are_not_domains(token):
    assert inventory._clean_domain(token) is None


@pytest.mark.parametrize("token", ["asiaone.com", "straitstimes.com.sg", "japantimes.co.jp", "wiki.gg", "time.is", "WWW.Reddit.com"])
def test_real_sites_are_domains(token):
    assert inventory._clean_domain(token) is not None


# ── lookup, market gate, sitelist fields ─────────────────────────────

SITES = {
    "asiaone.com": {"domain": "asiaone.com", "name": "AsiaOne", "bidRequests": 30500000, "editorial": True,
                    "category": "general news", "region": "singapore", "niches": ["travel", "lifestyle"],
                    "creative": {"Banner": {"bidRequests": 30000000, "cpmMin": 0.42, "cpmMax": 0.84},
                                 "Video": {"bidRequests": 500000, "cpmMin": 1.1, "cpmMax": 2.2}}},
    "sethlui.com": {"domain": "sethlui.com", "name": "Seth Lui", "bidRequests": 4000000, "editorial": True,
                    "category": "food & dining", "region": "singapore", "niches": ["food"],
                    "creative": {"Banner": {"bidRequests": 4000000, "cpmMin": 0.16, "cpmMax": 0.32}}},
    "poki.com": {"domain": "poki.com", "name": "Poki", "bidRequests": 3000000, "editorial": False,
                 "creative": {"Banner": {"bidRequests": 3000000, "cpmMin": 0.44, "cpmMax": 0.88}}},
}


def test_lookup_matches_subdomains():
    assert inventory.lookup("www.asiaone.com", SITES)["domain"] == "asiaone.com"
    assert inventory.lookup("lifestyle.asiaone.com", SITES)["domain"] == "asiaone.com"
    assert inventory.lookup("notasiaone.com", SITES) is None


def test_market_gate_is_singapore_only():
    assert inventory.applies_to("Singapore")
    assert inventory.applies_to("SG")
    assert inventory.applies_to("singapore and malaysia")
    assert not inventory.applies_to("Philippines")
    assert not inventory.applies_to(None)


async def test_sitelist_marks_inventory_sources_buyable_only_in_singapore(monkeypatch):
    from app.models import Publisher
    monkeypatch.setattr(inventory, "load_sites", lambda: SITES)
    pubs = [Publisher(name="asiaone.com", type="News", mentions=2), Publisher(name="other.example", type="Blog", mentions=1)]
    reach = {"asiaone.com": 3, "other.example": 1}
    sg = grounding._build_sitelist(pubs, reach, 4, use_inventory=True)
    assert sg[0].buyable and sg[0].cpm == "0.42 - 2.20" and sg[0].inventory == ["Banner", "Video"]
    assert "30,500,000" in sg[0].availability
    assert [e.score for e in sg] == [75, 25]  # share of buyer questions that surfaced it
    assert not sg[1].buyable and sg[1].cpm == "—"
    elsewhere = grounding._build_sitelist(pubs, reach, 4, use_inventory=False)
    assert not any(e.buyable for e in elsewhere)


def test_user_sites_on_hosting_platforms_are_not_buyable_as_the_platform():
    sites = {"tumblr.com": {"domain": "tumblr.com"}, "yahoo.com": {"domain": "yahoo.com"}}
    assert inventory.lookup("someone.tumblr.com", sites) is None
    assert inventory.lookup("tumblr.com", sites)["domain"] == "tumblr.com"
    assert inventory.lookup("sg.news.yahoo.com", sites)["domain"] == "yahoo.com"
