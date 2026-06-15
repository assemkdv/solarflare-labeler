import pytest
from solarflare_labeler.builder import DatasetBuilder
from solarflare_labeler.strategies import BinaryThresholdStrategy


def _write_valid_catalog(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,start,peak,end,class,active_region\n"
        "2024-01-01,2024-01-01T00:20:00,2024-01-01T00:30:00,2024-01-01T00:40:00,M2.3,4456\n"
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
