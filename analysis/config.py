"""
Project-wide configuration: temporal windows, cities, sources, and folder paths.
"""

from datetime import date

# ---------------------------------------------------------------------------
# Temporal windows — event year (2026)
# ---------------------------------------------------------------------------

EVENT_WINDOWS: dict[str, tuple[date, date]] = {
    "pre":     (date(2026, 4,  1), date(2026, 5, 20)),
    "during":  (date(2026, 5, 21), date(2026, 5, 24)),
    "post":    (date(2026, 5, 25), date(2026, 6, 15)),
}

# Baseline windows use the same calendar spans shifted to 2025.
BASELINE_WINDOWS: dict[str, tuple[date, date]] = {
    "baseline_pre":     (date(2025, 4,  1), date(2025, 5, 20)),
    "baseline_during":  (date(2025, 5, 21), date(2025, 5, 24)),
    "baseline_post":    (date(2025, 5, 25), date(2025, 6, 15)),
}

# ---------------------------------------------------------------------------
# Cities
# ---------------------------------------------------------------------------

TARGET_CITY: str = "Cagliari"
CONTROL_CITY: str = "Olbia"
CITIES: list[str] = [TARGET_CITY, CONTROL_CITY]

# ---------------------------------------------------------------------------
# Scrape sources
# ---------------------------------------------------------------------------

SOURCES: list[str] = ["booking", "airbnb", "tripadvisor", "google"]

# ---------------------------------------------------------------------------
# Folder paths  (relative to project root)
# ---------------------------------------------------------------------------

RAW_DATA_DIR: str       = "data/raw/"
PROCESSED_DATA_DIR: str = "data/processed/"
BASELINES_DATA_DIR: str = "data/baselines/"
RESULTS_DIR: str        = "results/"
