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
    input_type : str
        Type of solar image input (e.g. "AIA_193").
    cadence : int
        Cadence in minutes between consecutive images in the index.
    sliding_window : int
        Number of consecutive images to group per sample (1 = single-image).
    strategy : BinaryThresholdStrategy | MaxFlareStrategy
        Labeling strategy that converts a list of FlareEvents to a label.
    """

    def __init__(self, prediction_window, input_type, cadence, sliding_window, strategy):
        self.prediction_window = prediction_window
        self.input_type = input_type
        self.cadence = cadence
        self.sliding_window = sliding_window
        self.strategy = strategy

    def build(self, image_index_path: str | Path, event_catalog_path: str | Path) -> pd.DataFrame:
        """
        Build a labeled dataset.

        Returns a DataFrame with columns:
            timestamp  - image timestamp (pd.Timestamp)
            label      - label produced by the strategy
        """
        image_index_path = Path(image_index_path)
        timestamps = pd.read_csv(image_index_path, parse_dates=["timestamp"])["timestamp"]

        matcher = EventMatcher(event_catalog_path)

        records = []
        for ts in timestamps:
            flares = matcher.query(ts, self.prediction_window)
            label = self.strategy.label(flares)
            records.append({"timestamp": ts, "label": label})

        return pd.DataFrame(records)
