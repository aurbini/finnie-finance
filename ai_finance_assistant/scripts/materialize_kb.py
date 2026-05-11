"""
Generate deterministic markdown KB articles under data/articles.

Run from checkout:
    python scripts/materialize_kb.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "articles"

TEMPLATE = """\
---
title: "{title}"
category: {category}
source_id: {slug}
---

# {title}

## Snapshot

{snapshot}

## What it means for beginners

{explain}

## Common pitfalls & myths

{pitfalls}

## Questions for self study

- Compare two timelines (5 years vs 30 years) and list how `{slug}` framing changes urgency and risk wording.
- Map one claim from this lesson to neutral sources (SEC Investor.gov, FINRA, IRS Publication landing pages—not clickbait).

## Remember

{remember}
"""


_TOPICS: list[tuple[str, str, str]] = [
    ("what-is-a-stock", "What Is a Stock?", "investing_basics"),
    ("bonds-explained-simply", "How Bonds Basically Work", "investing_basics"),
    ("index-funds-explained", "Index Funds Explained", "portfolio"),
    ("active-vs-passive-intro", "Active vs Passive Investing (Introductory)", "investing_basics"),
    ("diversification-101", "Diversification 101", "portfolio"),
    ("compound-interest-intuition", "Compound Interest Without the Hype", "investing_basics"),
    ("time-in-vs-timing", "Time in Market vs Timing the Market", "behavioral"),
    ("volatility-explained-plainly", "Volatility Explained Plainly", "markets"),
    ("risk-vs-return-tradeoff", "Risk vs Reward Trade-offs", "portfolio"),
    ("asset-allocation-intro", "Asset Allocation Intro", "portfolio"),
    ("reading-a-fund-prospectus", "How to Read a Fund Prospectus (Lightweight Tour)", "investing_basics"),
    ("etf-vs-mutual-fund", "ETFs vs Mutual Funds (Retail Angle)", "investing_basics"),
    ("fractional-shares-explainer", "Fractional Shares Basics", "investing_basics"),
    ("custodian-and-broker-roles", "Custodians, Brokers, and Advisors (Big Picture)", "investing_basics"),
    ("order-types-explainer-intro", "Order Types for Beginners", "investing_basics"),
    ("margin-basics-cautionary", "Margin Basics (Education + Caution)", "investing_basics"),
    ("short-selling-concept-overview", "What Short Selling Conceptually Means", "markets"),
    ("options-intro-not-how-to", "Options Concepts for Literacy (Not Instructions)", "markets"),
    ("market-cap-explainer", "What Market Capitalization Signals (and Doesn't)", "markets"),
    ("dividends-basics-and-tax-notes", "Dividends: Basics Plus Tax Literacy Angles", "taxes"),
    ("stock-splits-reverse-splits-overview", "Splits and Reverse Splits Overview", "markets"),
    ("earnings-reports-beginner-scan", "Earnings Releases: What to Scan", "markets"),
    ("sector-and-industry-basics", "Sectors and Industries Intro", "markets"),
    ("macro-vs-micro-simple", "Macro vs Company Stories in Plain Speech", "markets"),
    ("inflation-and-purchasing-power", "Inflation and Purchasing Power", "investing_basics"),
    ("interest-rate-sensitivity-intro", "Why Interest Rates Ripple Through Assets", "markets"),
    ("recession-conversation-starters", "Recession Narratives Starter Pack", "behavioral"),
    ("behavioral-loss-aversion-intro", "Loss Aversion Basics", "behavioral"),
    ("mental-accounting-explainer", "Mental Accounting Explained", "behavioral"),
    ("anchoring-heuristics-beginners", "Anchoring Heuristic Intro", "behavioral"),
    ("overconfidence-financial-intro", "Overconfidence in Investing Intro", "behavioral"),
    ("confirmation-bias-in-news", "Confirmation Bias in Financial News", "behavioral"),
    ("dollar-cost-averaging-concept-only", "Dollar Cost Averaging Themes", "investing_basics"),
    ("lump-sum-vs-dca-studies-tone", "Lump Sum vs DCA Narrative Tone Calibration", "behavioral"),
    ("rebalancing-explainer", "Portfolio Rebalancing Basics", "portfolio"),
    ("tax-loss-harvesting-concept-intro", "Tax Loss Harvesting Literacy Only", "taxes"),
    ("wash-sale-concept-explainer", "Wash Sale Rule Concept Outline", "taxes"),
    ("qualified-vs-nonqualified-explainer", "Qualified vs Ordinary Tax Themes Intro", "taxes"),
    ("401k-structure-basics-educational-only", "401k Mechanics Basics Educational", "taxes"),
    ("roth-vs-traditional-explainer-intro", "Roth vs Traditional Shelter Concepts", "retirement"),
    ("hsa-educational-triple-tax-framing-with-caveats", "HSAs Framing with Caveats", "taxes"),
    ("ira-contribution-rules-outline", "IRA Topics Outline Educational", "retirement"),
    ("required-minimum-withdrawals-outline", "RMD Concepts Outline", "retirement"),
    ("social-security-basics-shape-only", "Social Security Literacy Starter", "retirement"),
    ("annuities-structure-not-a-recommendation", "Annuity Mechanics Intro Not Recommendation", "retirement"),
    ("target-date-funds-concept-explainer", "Target Date Mechanics Explained", "portfolio"),
    ("529-plans-shape-only-outline", "529 Outline Educational", "taxes"),
    ("estate-tax-basics-starter", "Estate Tax Starter Concepts", "taxes"),
    ("step-up-cost-basis-concept-explainer", "Step Up Basis Concepts Literacy", "taxes"),
    ("charitable-strategies-shape-only-overview", "Charitable Giving Strategies Outline", "taxes"),
    ("crypto-education-shape-only-intro", "Digital Assets Literacy Intro", "markets"),
    ("esg-reading-list-conversation", "ESG Conversations Literacy", "behavioral"),
    ("credit-scores-mortgages-lite", "Credit Scores and Mortgages Intro", "investing_basics"),
    ("emergency-fund-sizing-conversation", "Emergency Fund Sizing Discussion", "investing_basics"),
    ("debt-snowball-vs-avalanche-educational", "Debt Paydown Strategies Literacy", "investing_basics"),
    ("financial-plan-schema-educational", "Goal Planning Literacy Schema", "retirement"),
    ("liquidity-vs-volatility-basics", "Liquidity Compared to Volatility", "portfolio"),
]


def paragraphs(*parts: str) -> str:
    return "\n\n".join(textwrap.fill(p.strip(), width=94) for p in parts)


def corrected_md(slug: str, title: str, category: str) -> str:
    snapshot = paragraphs(
        f"This `{slug}` entry explains `{title.lower()}` plainly for newcomers.",
        "It avoids live prices because education should outlast ticker snapshots.",
    )
    explain = paragraphs(
        "Anchor your thinking on diversification, horizons, disclosures, disclaimers—not predictions.",
        "Different accounts and countries change outcomes; widen reading beyond summaries.",
        "Stories about extremes are selectively memorable; broaden data before acting.",
    )
    pitfalls = "\n".join(
        [
            "- Treating anecdotes as benchmarks.",
            "- Using leverage casually without understanding liquidation paths.",
            "- Confusing financial entertainment with fiduciary advice.",
            "- Ignoring tax nuance tied to geography and filings.",
            "- Believing jargon equals correctness.",
        ]
    )
    remember = paragraphs(
        "Illustrative only; rely on regulators and pros for personalization."
    )

    return TEMPLATE.format(
        title=title.replace('"', "'"),
        category=category,
        slug=slug,
        snapshot=snapshot,
        explain=explain,
        pitfalls=pitfalls,
        remember=remember,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for slug, title, category in _TOPICS:
        (OUT / f"{slug}.md").write_text(corrected_md(slug, title, category), encoding="utf-8")
    print(f"Wrote {len(_TOPICS)} markdown articles -> {OUT}")


if __name__ == "__main__":
    main()
