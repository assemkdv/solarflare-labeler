import pandas as pd
from pathlib import Path
from .events import EventMatcher


class DatasetBuilder:
    """
    Maps image timestamps to flare labels using a configurable strategy.

    Parameters
    ----------
    prediction_window : int
        Hours after each image timestamp to search for flares.
    strategy : BinaryThresholdStrategy | MaxFlareStrategy
        Labeling strategy that converts a list of FlareEvents to a label.
    """

    def __init__(self, prediction_window, strategy):
        self.prediction_window = prediction_window
        self.strategy = strategy

    def build(self, image_index_path: str | Path, event_catalog_path: str | Path) -> pd.DataFrame:
        """
        Build a labeled dataset.

        Returns a DataFrame with columns:
            timestamp  - image timestamp (pd.Timestamp)
            label      - label produced by the strategy

        An image index with a header row but zero data rows is valid and
        produces an empty result with the same (timestamp, label) columns.
        A completely empty image index (no header) raises ValueError, as
        does an image index missing the required timestamp column.
        """
        image_index_path = Path(image_index_path)

        try:
            index_df = pd.read_csv(image_index_path)
        except pd.errors.EmptyDataError:
            raise ValueError(
                f"Image index at {image_index_path} is empty: no header row / columns found."
            ) from None

        if "timestamp" not in index_df.columns:
            raise ValueError(
                f"Image index at {image_index_path} is missing required column: timestamp"
            )

        timestamps = pd.to_datetime(index_df["timestamp"])

        matcher = EventMatcher(event_catalog_path)

        records = []
        for ts in timestamps:
            flares = matcher.query(ts, self.prediction_window)
            label = self.strategy.label(flares)
            records.append({"timestamp": ts, "label": label})

        if not records:
            return pd.DataFrame(columns=["timestamp", "label"])
        return pd.DataFrame(records)
