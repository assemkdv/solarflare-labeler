import pandas as pd
import pytest
from solarflare_labeler.builder import DatasetBuilder
from solarflare_labeler.strategies import BinaryThresholdStrategy, MaxFlareStrategy


def _write_valid_catalog(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,start,peak,end,class,active_region\n"
        "2024-01-01,0020,0030,0040,M2.3,4456\n"
    )
    return catalog_path


def _write_two_flare_catalog(tmp_path):
    # Flare A: M5.0 at 2024-01-01T02:00 (qualifies M threshold, flux=5e-5)
    # Flare B: C3.0 at 2024-01-03T02:00 (does not qualify, flux=3e-6)
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,start,peak,end,class,active_region\n"
        "2024-01-01,0150,0200,0210,M5.0,4456\n"
        "2024-01-03,0150,0200,0210,C3.0,4456\n"
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


# ── Sequence labeling tests ─────────────────────────────────────────────


def test_sequence_length_one_matches_single_image_schema(tmp_path):
    catalog_path = _write_two_flare_catalog(tmp_path)
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp\n"
        "2024-01-01T00:00:00\n"
        "2024-01-03T00:00:00\n"
    )

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(), sequence_length=1, stride=1
    )
    result = builder.build(index_path, catalog_path)

    assert list(result.columns) == ["timestamp", "label"]
    assert result["label"].tolist() == [1, 0]


def test_sequences_overlapping_with_stride_one(tmp_path):
    catalog_path = _write_valid_catalog(tmp_path)
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp\n"
        "2024-01-01T08:00:00\n"
        "2024-01-01T08:12:00\n"
        "2024-01-01T08:24:00\n"
        "2024-01-01T08:36:00\n"
        "2024-01-01T08:48:00\n"
    )

    builder = DatasetBuilder(
        prediction_window=24,
        strategy=BinaryThresholdStrategy(),
        sequence_length=3,
        stride=1,
        cadence_minutes=12,
    )
    result = builder.build(index_path, catalog_path)

    assert result["sequence_start"].tolist() == [
        pd.Timestamp("2024-01-01T08:00:00"),
        pd.Timestamp("2024-01-01T08:12:00"),
        pd.Timestamp("2024-01-01T08:24:00"),
    ]
    assert result["sequence_end"].tolist() == [
        pd.Timestamp("2024-01-01T08:24:00"),
        pd.Timestamp("2024-01-01T08:36:00"),
        pd.Timestamp("2024-01-01T08:48:00"),
    ]
    assert result["timestamps"].iloc[0] == [
        pd.Timestamp("2024-01-01T08:00:00"),
        pd.Timestamp("2024-01-01T08:12:00"),
        pd.Timestamp("2024-01-01T08:24:00"),
    ]
    assert result["n_images"].tolist() == [3, 3, 3]


def test_sequences_non_overlapping_with_stride_equal_length(tmp_path):
    catalog_path = _write_valid_catalog(tmp_path)
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp\n"
        "2024-01-01T08:00:00\n"
        "2024-01-01T08:12:00\n"
        "2024-01-01T08:24:00\n"
        "2024-01-01T08:36:00\n"
        "2024-01-01T08:48:00\n"
        "2024-01-01T09:00:00\n"
    )

    builder = DatasetBuilder(
        prediction_window=24,
        strategy=BinaryThresholdStrategy(),
        sequence_length=3,
        stride=3,
        cadence_minutes=12,
    )
    result = builder.build(index_path, catalog_path)

    assert result["sequence_start"].tolist() == [
        pd.Timestamp("2024-01-01T08:00:00"),
        pd.Timestamp("2024-01-01T08:36:00"),
    ]
    assert result["sequence_end"].tolist() == [
        pd.Timestamp("2024-01-01T08:24:00"),
        pd.Timestamp("2024-01-01T09:00:00"),
    ]


def test_sequence_prediction_window_starts_at_final_image(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,start,peak,end,class,active_region\n"
        "2024-01-01,0824,0824,0829,M1.0,4456\n"
    )
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp\n"
        "2024-01-01T08:00:00\n"
        "2024-01-01T08:12:00\n"
        "2024-01-01T08:24:00\n"
    )

    builder = DatasetBuilder(
        prediction_window=0.2,  # 12 minutes
        strategy=BinaryThresholdStrategy(),
        sequence_length=3,
        stride=1,
        cadence_minutes=12,
    )
    result = builder.build(index_path, catalog_path)

    assert len(result) == 1
    assert result.iloc[0]["sequence_end"] == pd.Timestamp("2024-01-01T08:24:00")
    # If matching incorrectly used sequence_start (08:00) instead of
    # sequence_end (08:24), this M1.0 flare peaking exactly at 08:24 would
    # fall outside a 12-minute window measured from 08:00, and label would
    # come out 0 instead of 1.
    assert result.iloc[0]["label"] == 1


def test_sequence_boundary_flare_excluded_at_far_edge(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,start,peak,end,class,active_region\n"
        "2024-01-01,0900,0900,0905,X1.0,4456\n"
    )
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp\n"
        "2024-01-01T07:48:00\n"
        "2024-01-01T08:00:00\n"
    )

    builder = DatasetBuilder(
        prediction_window=1,
        strategy=BinaryThresholdStrategy(),
        sequence_length=2,
        stride=1,
        cadence_minutes=12,
    )
    result = builder.build(index_path, catalog_path)

    assert len(result) == 1
    assert result.iloc[0]["sequence_end"] == pd.Timestamp("2024-01-01T08:00:00")
    # X1.0 peaks exactly at sequence_end + prediction_window (09:00) and
    # must be excluded by the half-open window; if it were wrongly
    # included, label would be 1 instead of 0.
    assert result.iloc[0]["label"] == 0


def test_sequences_insufficient_rows_returns_empty(tmp_path):
    catalog_path = _write_valid_catalog(tmp_path)
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp\n"
        "2024-01-01T08:00:00\n"
        "2024-01-01T08:12:00\n"
    )

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(), sequence_length=5, stride=1
    )
    result = builder.build(index_path, catalog_path)

    assert list(result.columns) == ["sequence_start", "sequence_end", "timestamps", "n_images", "label"]
    assert len(result) == 0


def test_sequence_length_must_be_positive():
    with pytest.raises(ValueError) as excinfo:
        DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy(), sequence_length=0)

    assert "sequence_length" in str(excinfo.value)


def test_stride_must_be_positive():
    with pytest.raises(ValueError) as excinfo:
        DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy(), stride=0)

    assert "stride" in str(excinfo.value)


def test_cadence_minutes_must_be_positive():
    with pytest.raises(ValueError) as excinfo:
        DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy(), cadence_minutes=0)

    assert "cadence_minutes" in str(excinfo.value)


def test_sequences_require_sorted_timestamps(tmp_path):
    catalog_path = _write_valid_catalog(tmp_path)
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp\n"
        "2024-01-01T08:12:00\n"
        "2024-01-01T08:00:00\n"
        "2024-01-01T08:24:00\n"
    )

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(), sequence_length=2, stride=1
    )

    with pytest.raises(ValueError) as excinfo:
        builder.build(index_path, catalog_path)

    assert "sorted" in str(excinfo.value).lower()


def test_sequences_reject_duplicate_timestamps(tmp_path):
    catalog_path = _write_valid_catalog(tmp_path)
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp\n"
        "2024-01-01T08:00:00\n"
        "2024-01-01T08:00:00\n"
        "2024-01-01T08:12:00\n"
    )

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(), sequence_length=2, stride=1
    )

    with pytest.raises(ValueError) as excinfo:
        builder.build(index_path, catalog_path)

    assert "duplicate" in str(excinfo.value).lower()


def test_sequences_skip_across_cadence_gap(tmp_path):
    catalog_path = _write_valid_catalog(tmp_path)
    index_path = tmp_path / "index.csv"
    # Regular 12-minute cadence except for a large gap between 08:24 and 09:30.
    index_path.write_text(
        "timestamp\n"
        "2024-01-01T08:00:00\n"
        "2024-01-01T08:12:00\n"
        "2024-01-01T08:24:00\n"
        "2024-01-01T09:30:00\n"
        "2024-01-01T09:42:00\n"
    )

    builder = DatasetBuilder(
        prediction_window=24,
        strategy=BinaryThresholdStrategy(),
        sequence_length=2,
        stride=1,
        cadence_minutes=12,
    )
    result = builder.build(index_path, catalog_path)

    # Candidate windows are [08:00,08:12], [08:12,08:24], [08:24,09:30]
    # (66-minute gap -- must be skipped), [09:30,09:42].
    assert result["sequence_start"].tolist() == [
        pd.Timestamp("2024-01-01T08:00:00"),
        pd.Timestamp("2024-01-01T08:12:00"),
        pd.Timestamp("2024-01-01T09:30:00"),
    ]


def test_sequences_work_with_max_flare_strategy(tmp_path):
    catalog_path = _write_two_flare_catalog(tmp_path)
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp\n"
        "2024-01-01T00:00:00\n"
        "2024-01-01T01:00:00\n"
        "2024-01-03T00:00:00\n"
        "2024-01-03T01:00:00\n"
    )

    builder = DatasetBuilder(
        prediction_window=24,
        strategy=MaxFlareStrategy(),
        sequence_length=2,
        stride=2,
    )
    result = builder.build(index_path, catalog_path)

    assert len(result) == 2
    assert result["label"].tolist() == [5e-5, 3e-6]


def test_sequences_preserve_extra_columns(tmp_path):
    catalog_path = _write_valid_catalog(tmp_path)
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp,image_path\n"
        "2024-01-01T08:00:00,img_0800.fits\n"
        "2024-01-01T08:12:00,img_0812.fits\n"
        "2024-01-01T08:24:00,img_0824.fits\n"
    )

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(), sequence_length=2, stride=1
    )
    result = builder.build(index_path, catalog_path)

    assert "image_path" in result.columns
    assert result.iloc[0]["image_path"] == ["img_0800.fits", "img_0812.fits"]
    assert result.iloc[1]["image_path"] == ["img_0812.fits", "img_0824.fits"]
