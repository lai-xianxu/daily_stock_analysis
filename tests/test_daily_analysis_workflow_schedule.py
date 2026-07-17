"""Static guarantees for the post-close GitHub Actions workflow."""

from pathlib import Path


WORKFLOW_PATH = (
    Path(__file__).resolve().parent.parent
    / ".github"
    / "workflows"
    / "00-daily-analysis.yml"
)


def test_daily_analysis_starts_at_1530_without_random_delay() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "cron: '30 7 * * 1-5'" in workflow
    assert "RANDOM % 60" not in workflow
    assert "ANALYSIS_TIMEOUT_MINUTES || '90'" in workflow


def test_stock_analysis_always_keeps_market_review() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "- stocks-only" not in workflow
    assert "python main.py --no-market-review" not in workflow
    assert "MARKET_REVIEW_ENABLED: 'true'" in workflow
