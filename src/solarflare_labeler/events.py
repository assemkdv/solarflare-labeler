import pandas as pd
from pathlib import Path
from dataclasses import dataclass


# A simple container that holds the details of one flare event
# Think of it like a row from the NOAA catalog, but as a Python object
@dataclass
class FlareEvent:
    peak_time: pd.Timestamp      # when the flare was at its strongest
    goes_class: str              # flare class letter + number, e.g. "M2.3"
    start_time: pd.Timestamp     # when the flare started
    active_region: str | None    # which region of the sun it came from (can be empty)


class EventMatcher:
    """
    Given an image timestamp, finds all flares that happened
    within a certain number of hours after that image was taken.
    """

    def __init__(self, catalog_path: str | Path):
        # Load the catalog once when the object is created
        # so we don't re-read the file every time we query
        self._catalog = self._load_catalog(catalog_path)

    def _load_catalog(self, path: str | Path) -> pd.DataFrame:
        # Read the CSV and automatically parse date columns into datetime objects
        df = pd.read_csv(path)
        df = df.rename(columns={"peak": "peak_time", "start": "start_time", "class": "goes_class"})
        df["peak_time"] = pd.to_datetime(df["peak_time"])
        df["start_time"] = pd.to_datetime(df["start_time"])

        # Sort by peak_time so we can do fast lookups later
        df = df.sort_values("peak_time").reset_index(drop=True)

        print(f"Loaded {len(df)} flare events")
        return df

    def query(
        self,
        image_time: pd.Timestamp,
        prediction_window_hours: int,
    ) -> list[FlareEvent]:
        """
        Returns all flares whose peak falls inside the prediction window.

        Example: image taken at 10:00, window = 24 hours
        → returns all flares with peak_time between 10:00 and 10:00 next day
        """

        # Calculate the end of the prediction window
        window_end = image_time + pd.Timedelta(hours=prediction_window_hours)

        # Create a True/False mask for every row in the catalog
        # True = this flare's peak falls inside our window
        mask = (
            (self._catalog["peak_time"] >= image_time) &
            (self._catalog["peak_time"] < window_end)
        )

        # Convert the matching rows into FlareEvent objects and return them
        # If no flares match, this returns an empty list → label will be 0
        return [
            FlareEvent(
                peak_time=row["peak_time"],
                goes_class=row["goes_class"],
                start_time=row["start_time"],
                active_region=row.get("active_region"),
            )
            for _, row in self._catalog[mask].iterrows()
        ]