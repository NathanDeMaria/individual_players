from datetime import datetime, timezone


_SEASON_TURNOVER_MONTH = 7


def get_current_year() -> int:
    """Get the current NCAABB season's 'first' year."""
    now = datetime.now(timezone.utc)
    return now.year - 1 if now.month < _SEASON_TURNOVER_MONTH else now.year
