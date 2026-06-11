"""
Utility functions for date parsing and temporal-window assignment.
"""

import re
from datetime import date, datetime
from typing import Optional

from analysis.config import EVENT_WINDOWS, BASELINE_WINDOWS

# ---------------------------------------------------------------------------
# Italian month name → month number
# ---------------------------------------------------------------------------

_IT_MONTHS: dict[str, int] = {
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4,
    "maggio": 5, "giugno": 6, "luglio": 7, "agosto": 8,
    "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12,
}

# Matches strings like "1º aprile 2025", "15 maggio 2026", "3° giugno 2025"
_IT_DATE_RE = re.compile(
    r"(\d{1,2})[°º°]?\s+([a-zà-ü]+)\s+(\d{4})",
    re.IGNORECASE,
)


def parse_italian_date(date_str: str) -> Optional[datetime]:
    """
    Parse a date string written in Italian into a datetime object.

    Supported format: "<day>[º°]? <month_name> <year>"
    Examples: "1º aprile 2025", "15 maggio 2026", "3 giugno 2025"

    Returns None if the string cannot be parsed.
    """
    if not date_str or not isinstance(date_str, str):
        return None

    match = _IT_DATE_RE.search(date_str.strip().lower())
    if not match:
        return None

    day_str, month_str, year_str = match.groups()
    month_num = _IT_MONTHS.get(month_str)
    if month_num is None:
        return None

    try:
        return datetime(int(year_str), month_num, int(day_str))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Temporal window assignment
# ---------------------------------------------------------------------------

# Label returned when a date falls outside every defined window.
_OUTSIDE_WINDOW = "outside"

# Combined lookup: event windows first, then baseline windows.
_ALL_WINDOWS: dict[str, tuple[date, date]] = {**EVENT_WINDOWS, **BASELINE_WINDOWS}


def assign_temporal_window(review_date: datetime | date, year: int) -> str:
    """
    Return the temporal-window label for *review_date*.

    For *year* == 2026, the event windows ("pre", "during", "post") are tested.
    For *year* == 2025, the baseline windows ("baseline_pre", …) are tested.
    Any other year returns "outside".

    Parameters
    ----------
    review_date:
        The date of the review (datetime or date object).
    year:
        The reference year used to select the correct set of windows.
        Must match the year in review_date; callers are responsible for
        consistency.

    Returns
    -------
    str
        One of: "pre", "during", "post",
                "baseline_pre", "baseline_during", "baseline_post",
                "outside".
    """
    if isinstance(review_date, datetime):
        review_date = review_date.date()

    if year == 2026:
        windows = EVENT_WINDOWS
    elif year == 2025:
        windows = BASELINE_WINDOWS
    else:
        return _OUTSIDE_WINDOW

    for label, (start, end) in windows.items():
        if start <= review_date <= end:
            return label

    return _OUTSIDE_WINDOW
