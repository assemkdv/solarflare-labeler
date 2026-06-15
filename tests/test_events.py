import pandas as pd
import pytest
from solarflare_labeler.events import EventMatcher


def test_query_normalizes_missing_active_region_to_none():
    # pandas represents a missing active_region cell as float NaN once the
    # column contains any missing values — this must come out as None.
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime(["2024-01-01 00:30"]),
        "start_time": pd.to_datetime(["2024-01-01 00:20"]),
        "goes_class": ["M2.3"],
        "active_region": [float("nan")],
    })

    results = matcher.query(
        image_time=pd.Timestamp("2024-01-01 00:00"),
        prediction_window_hours=1,
    )

    assert results[0].active_region is None


def test_query_preserves_non_empty_active_region():
    # A present active_region value in a column that also has missing cells
    # arrives from pandas as a float (e.g. 4456.0) — it must be preserved
    # as a proper string ("4456"), not dropped or left as a float.
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime(["2024-01-01 00:30"]),
        "start_time": pd.to_datetime(["2024-01-01 00:20"]),
        "goes_class": ["M2.3"],
        "active_region": [4456.0],
    })

    results = matcher.query(
        image_time=pd.Timestamp("2024-01-01 00:00"),
        prediction_window_hours=1,
    )

    assert results[0].active_region == "4456"
    assert isinstance(results[0].active_region, str)


def test_load_catalog_does_not_print(tmp_path, capsys):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,start,peak,end,class,active_region\n"
        "2024-01-01,2024-01-01T00:20:00,2024-01-01T00:30:00,2024-01-01T00:40:00,M2.3,4456\n"
    )

    EventMatcher(catalog_path)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_load_catalog_missing_one_required_column(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,start,peak,end,class\n"
        "2024-01-01,2024-01-01T00:20:00,2024-01-01T00:30:00,2024-01-01T00:40:00,M2.3\n"
    )

    with pytest.raises(ValueError) as excinfo:
        EventMatcher(catalog_path)

    message = str(excinfo.value)
    assert "active_region" in message
    assert str(catalog_path) in message


def test_load_catalog_missing_multiple_required_columns(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,peak,class\n"
        "2024-01-01,2024-01-01T00:30:00,M2.3\n"
    )

    with pytest.raises(ValueError) as excinfo:
        EventMatcher(catalog_path)

    message = str(excinfo.value)
    assert "start" in message
    assert "end" in message
    assert "active_region" in message


def test_load_catalog_completely_empty_file_raises(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text("")

    with pytest.raises(ValueError) as excinfo:
        EventMatcher(catalog_path)

    assert "empty" in str(excinfo.value).lower()


def test_load_catalog_header_only_is_valid_and_empty(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text("date,start,peak,end,class,active_region\n")

    matcher = EventMatcher(catalog_path)
    results = matcher.query(pd.Timestamp("2024-01-01"), prediction_window_hours=24)

    assert results == []
