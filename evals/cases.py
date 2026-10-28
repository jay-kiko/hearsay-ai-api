"""Gold cases for the live eval (evals/run.py). Three Singapore-market brands
from different industries, so a change that helps one kind of brand (e.g. travel)
can't quietly regress another. Add a case when a client reports a new failure.

`names` are the brand/competitor strings that must NOT appear in generated
personas or prompts (neutrality check in evals/metrics.py).
"""

CASES = [
    dict(
        name="disney_cruise",
        brand="Disney Cruise Line",
        aliases=["Disney Cruise Line", "Disney Cruise", "DCL"],
        industry="Cruises",
        market="Singapore",
        buyer="Families and couples in Singapore choosing a cruise holiday",
        summary=(
            "Disney Cruise Line is a family-focused cruise line operated by Disney, known for "
            "Disney-themed ships, character experiences and kids' clubs."
        ),
        comps=[
            ("Royal Caribbean", ["Royal Caribbean"]),
            ("Resorts World Cruises", ["Resorts World Cruises", "Genting Dream"]),
            ("Celebrity Cruises", ["Celebrity Cruises"]),
        ],
        names=["disney", "royal caribbean", "resorts world", "genting", "celebrity"],
    ),
    dict(
        name="love_bonito",
        brand="Love, Bonito",
        aliases=["Love, Bonito", "Love Bonito"],
        industry="Women's fashion retail",
        market="Singapore",
        buyer="Women in Singapore choosing where to buy everyday and workwear clothing",
        summary=(
            "Love, Bonito is a Singapore-founded direct-to-consumer women's fashion label selling "
            "workwear and casual wear online and in stores."
        ),
        comps=[("Zalora", ["Zalora"]), ("Cotton On", ["Cotton On"]), ("Uniqlo", ["Uniqlo"])],
        names=["love, bonito", "love bonito", "zalora", "cotton on", "uniqlo"],
    ),
    dict(
        name="xero",
        brand="Xero",
        aliases=["Xero"],
        industry="Small business accounting software",
        market="Singapore",
        buyer="Singapore small business owners and bookkeepers choosing cloud accounting software",
        summary=(
            "Xero is cloud accounting software for small businesses, covering invoicing, bank "
            "reconciliation, payroll integrations and reporting."
        ),
        comps=[("QuickBooks", ["QuickBooks"]), ("MYOB", ["MYOB"]), ("Zoho Books", ["Zoho Books"])],
        names=["xero", "quickbooks", "myob", "zoho"],
    ),
]
