"""Converting governance capacity into the false-alarm budget (decision D2)."""

DAYS_PER_YEAR = 365.25


def window_days(cases_per_window: int, patients_per_week: float) -> float:
    """Calendar length of one monitoring window, in days, at a case volume.

    Raises:
        ValueError: When the volume is not positive.
    """
    if patients_per_week <= 0:
        raise ValueError("patients_per_week must be positive")
    return cases_per_window / patients_per_week * 7.0


def false_alarms_per_1000_windows(reviews_per_year: float, window_days: float, tools_sharing: float = 1.0) -> float:
    """Alert episodes per 1,000 monitored windows for one tool, given a committee's capacity for false-alarm reviews per year.

    A committee that shares its capacity across `tools_sharing` tools can give each tool only its share.

    Raises:
        ValueError: When the window length or the number of tools is not positive.
    """
    if window_days <= 0 or tools_sharing <= 0:
        raise ValueError("window_days and tools_sharing must be positive")
    windows_per_year = DAYS_PER_YEAR / window_days
    return reviews_per_year / tools_sharing / windows_per_year * 1000.0
