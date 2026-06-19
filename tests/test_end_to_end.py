from pathlib import Path

import pandas as pd

from solarflare_labeler.builder import DatasetBuilder
from solarflare_labeler.strategies import BinaryThresholdStrategy, MaxFlareStrategy

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CATALOG_PATH = FIXTURES_DIR / "catalog.csv"
IMAGE_INDEX_PATH = FIXTURES_DIR / "image_index.csv"

# tests/fixtures/catalog.csv has four flares:
#   C3.0 peak 2024-02-01T12:00:00  (flux=3e-6)
#   X1.0 peak 2024-02-02T00:00:00  (flux=1e-4)  <- exactly on a 24h window end
#   C1.0 peak 2024-02-05T06:00:00  (flux=1e-6)
#   M4.0 peak 2024-02-05T18:00:00  (flux=4e-5)
#
# tests/fixtures/image_index.csv has three timestamps, each with a
# prediction_window of 24 hours:
#   2024-02-01T00:00:00 -> window [2024-02-01T00:00, 2024-02-02T00:00)
#       Only C3.0 is inside. X1.0 peaks exactly at the window end and must
#       be excluded by the half-open interval.
#   2024-02-05T00:00:00 -> window [2024-02-05T00:00, 2024-02-06T00:00)
#       Both C1.0 and M4.0 are inside; M4.0 qualifies the binary threshold.
#   2024-02-10T00:00:00 -> window [2024-02-10T00:00, 2024-02-11T00:00)
#       No flares fall inside; the window is empty.


def test_end_to_end_binary_threshold_strategy():
    builder = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy())
    result = builder.build(IMAGE_INDEX_PATH, CATALOG_PATH)

    assert list(result.columns) == ["timestamp", "label"]
    assert result["timestamp"].tolist() == [
        pd.Timestamp("2024-02-01T00:00:00"),
        pd.Timestamp("2024-02-05T00:00:00"),
        pd.Timestamp("2024-02-10T00:00:00"),
    ]
    assert result["label"].tolist() == [0, 1, 0]


def test_end_to_end_max_flare_strategy():
    builder = DatasetBuilder(prediction_window=24, strategy=MaxFlareStrategy())
    result = builder.build(IMAGE_INDEX_PATH, CATALOG_PATH)

    # Row 1: max is C3.0's 3e-6 — X1.0's 1e-4 must be excluded since its
    # peak lands exactly on the window boundary. If the boundary were
    # mistakenly inclusive, this would be 1e-4 instead.
    # Row 2: max of C1.0 (1e-6) and M4.0 (4e-5) is 4e-5.
    # Row 3: no flares in the window -> 0.0.
    assert result["label"].tolist() == [3e-6, 4e-5, 0.0]


def test_end_to_end_smoke_test_against_real_catalog(tmp_path):
    # Not the main correctness test — labels here aren't hardcoded since
    # the real development catalog can change over time. This just
    # confirms the full pipeline runs cleanly against actual data.
    repo_root = Path(__file__).parent.parent
    real_catalog = repo_root / "data" / "solar_events.csv"

    index_path = tmp_path / "smoke_index.csv"
    pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-06-06T00:00:00", "2026-06-10T00:00:00"]),
    }).to_csv(index_path, index=False)

    builder = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy())
    result = builder.build(index_path, real_catalog)

    assert list(result.columns) == ["timestamp", "label"]
    assert len(result) == 2
    assert set(result["label"].tolist()) <= {0, 1}
