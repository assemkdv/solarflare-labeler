import pandas as pd
import pytest
from solarflare_labeler.builder import DatasetBuilder
from solarflare_labeler.strategies import BinaryThresholdStrategy, MaxFlareStrategy


def _write_valid_catalog(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,start,peak,end,class,active_region\n"
        "2024-01-01,2024-01-01T00:20:00,2024-01-01T00:30:00,2024-01-01T00:40:00,M2.3,4456\n"
    )
    return catalog_path


def _write_two_flare_catalog(tmp_path):
    # Flare A: M5.0 at 2024-01-01T02:00 (qualifies M threshold, flux=5e-5)
    # Flare B: C3.0 at 2024-01-03T02:00 (does not qualify, flux=3e-6)
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,start,peak,end,class,active_region\n"
        "2024-01-01,2024-01-01T01:50:00,2024-01-01T02:00:00,2024-01-01T02:10:00,M5.0,4456\n"
        "2024-01-03,2024-01-03T01:50:00,2024-01-03T02:00:00,2024-01-03T02:10:00,C3.0,4456\n"
    )
    return catalog_path


def test_build_missing_timestamp_column_raises(tmp_path):
    index_path = tmp_path / "index.csv"
    index_path.write_text("not_timestamp\n2024-01-01T00:00:00\n")
    catalog_path = _write_valid_catalog(tmp_path)

    builder = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy())

    with pytest.raises(ValueError) as excinfo:
        builder.build(index_path, catalog_path)

    message = str(excinfo.value)
    assert "timestamp" in message
    assert str(index_path) in message


def test_build_completely_empty_image_index_raises(tmp_path):
    index_path = tmp_path / "index.csv"
    index_path.write_text("")
    catalog_path = _write_valid_catalog(tmp_path)

    builder = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy())

    with pytest.raises(ValueError) as excinfo:
        builder.build(index_path, catalog_path)

    assert "empty" in str(excinfo.value).lower()


def test_build_header_only_image_index_is_valid_and_empty(tmp_path):
    index_path = tmp_path / "index.csv"
    index_path.write_text("timestamp\n")
    catalog_path = _write_valid_catalog(tmp_path)

    builder = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy())
    result = builder.build(index_path, catalog_path)

    assert list(result.columns) == ["timestamp", "label"]
    assert len(result) == 0


def test_build_produces_expected_timestamp_and_label_columns(tmp_path):
    catalog_path = _write_two_flare_catalog(tmp_path)
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp\n"
        "2024-01-01T00:00:00\n"
        "2024-01-03T00:00:00\n"
    )

    builder = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy())
    result = builder.build(index_path, catalog_path)

    assert list(result.columns) == ["timestamp", "label"]
    assert result["timestamp"].tolist() == [
        pd.Timestamp("2024-01-01T00:00:00"),
        pd.Timestamp("2024-01-03T00:00:00"),
    ]
    # Jan 1 window catches Flare A (M5.0, qualifies); Jan 3 window catches
    # only Flare B (C3.0, does not qualify).
    assert result["label"].tolist() == [1, 0]


def test_build_preserves_image_timestamp_order(tmp_path):
    catalog_path = _write_two_flare_catalog(tmp_path)
    index_path = tmp_path / "index.csv"
    # Deliberately out of chronological order.
    index_path.write_text(
        "timestamp\n"
        "2024-01-03T00:00:00\n"
        "2024-01-01T00:00:00\n"
    )

    builder = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy())
    result = builder.build(index_path, catalog_path)

    assert result["timestamp"].tolist() == [
        pd.Timestamp("2024-01-03T00:00:00"),
        pd.Timestamp("2024-01-01T00:00:00"),
    ]
    assert result["label"].tolist() == [0, 1]


def test_build_works_with_max_flare_strategy(tmp_path):
    catalog_path = _write_two_flare_catalog(tmp_path)
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp\n"
        "2024-01-01T00:00:00\n"
        "2024-01-03T00:00:00\n"
    )

    builder = DatasetBuilder(prediction_window=24, strategy=MaxFlareStrategy())
    result = builder.build(index_path, catalog_path)

    assert result["label"].tolist() == [5e-5, 3e-6]
