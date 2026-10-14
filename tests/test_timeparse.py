import pandas as pd
import pytest

from backend.schema.timeparse import detect_time_format, has_time_of_day, parse_time


@pytest.mark.parametrize(
    "values,name,expected_fmt,first",
    [
        ([20250101, 20250215, 20251231], "sessionDate", "yyyymmdd", "2025-01-01"),
        ([1735689600, 1735776000], "created_ts", "unix_s", "2025-01-01"),
        ([1735689600000, 1735776000000], "started_at_ms", "unix_ms", "2025-01-01"),
        ([1735689600000000], "event_time", "unix_us", "2025-01-01"),
        (["2025-01-01T08:30:00", "2025-02-01T10:00:00"], "completed_at", "iso", "2025-01-01"),
        (["2025-01-01 08:30:00+02:00"], "occurred", "iso", "2025-01-01"),
        (["01/02/2025", "03/04/2025"], "day", "iso", None),
    ],
)
def test_detect_and_parse(values, name, expected_fmt, first):
    s = pd.Series(values)
    fmt, rate = detect_time_format(s, name=name)
    assert fmt == expected_fmt and rate >= 0.9
    parsed = parse_time(s, fmt)
    assert parsed.notna().all()
    if first:
        assert str(parsed.iloc[0].date()) == first


def test_timezone_is_normalised_to_utc():
    parsed = parse_time(pd.Series(["2025-01-01 01:00:00+02:00"]), "iso")
    assert parsed.iloc[0] == pd.Timestamp("2024-12-31 23:00:00")


def test_native_datetimes():
    s = pd.Series(pd.to_datetime(["2025-01-01", "2025-01-02"]))
    assert detect_time_format(s, "whatever")[0] == "native"


@pytest.mark.parametrize(
    "values,name",
    [
        ([1, 2, 3, 4], "user_id"),  # ids
        ([20250101, 20250102], "amount"),  # numbers without a time-like name
        (["free", "plus", "pro"], "plan"),  # categories
        ([12.5, 40.1], "duration_min"),  # measure whose name contains "min" but isn't time
    ],
)
def test_non_time_columns_are_rejected(values, name):
    assert detect_time_format(pd.Series(values), name=name)[0] is None


def test_has_time_of_day():
    assert not has_time_of_day(pd.Series(pd.to_datetime(["2025-01-01", "2025-01-02"])))
    assert has_time_of_day(pd.Series(pd.to_datetime(["2025-01-01 08:00", "2025-01-02 17:30"])))
