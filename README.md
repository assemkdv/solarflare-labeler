solarflare-labeler builds **labeled datasets for solar flare prediction**:
given a list of solar image timestamps and a catalog of flare events, it
looks forward in time from each image, checks whether a flare occurred
within a configurable prediction window, and produces a ready-to-train
table of `timestamp, label` pairs.

## Installation

Not yet published to PyPI. Install from source:

```bash
git clone https://github.com/assemkdv/solarflare-labeler.git
cd solarflare-labeler
pip install .
```

## Quickstart

```python
import pandas as pd
import solarflare_labeler as sfl

# A tiny image index: one column, "timestamp"
pd.DataFrame({
    "timestamp": pd.to_datetime(["2024-01-01T00:00:00", "2024-01-02T00:00:00"]),
}).to_csv("image_index.csv", index=False)

# A tiny flare catalog: date, start, peak, end, class, active_region
pd.DataFrame({
    "date": ["2024-01-01"],
    "start": ["2024-01-01T02:50:00"],
    "peak": ["2024-01-01T03:00:00"],
    "end": ["2024-01-01T03:10:00"],
    "class": ["M2.3"],
    "active_region": [12345],
}).to_csv("flare_catalog.csv", index=False)

strategy = sfl.BinaryThresholdStrategy()
builder = sfl.DatasetBuilder(prediction_window=24, strategy=strategy)

labeled = builder.build("image_index.csv", "flare_catalog.csv")
print(labeled)
```

```
   timestamp  label
0 2024-01-01      1
1 2024-01-02      0
```

The first image's 24-hour window catches the M2.3 flare peaking a few
hours later, so it's labeled `1`. The second image's window has no
flares in it, so it's labeled `0`.

## How it works (the pipeline)

```
image timestamp
      |
      v
[ EventMatcher ]   -> finds all flares within N hours after the image
      |
      v
[ Strategy ]       -> turns that list of flares into a single label
      |
      v
[ DatasetBuilder ] -> repeats for every image, returns a table: timestamp -> label
```

**`FlareClassifier`** — translates flare strength. Flares are named with a
letter + number ("M2.3", "X1.0"). Letters go A < B < C < M < X, and each
step is 10x stronger (a log scale). This class converts that text into a
real number (e.g. "M2.3" -> 2.3e-5) so flares can be compared, and answers
"is this flare strong enough to matter?" (`is_strong`, default threshold =
M-class).

**`EventMatcher`** — searches the catalog by time. Given an image
timestamp and a window (e.g. 24 hours), it returns every flare whose peak
fell inside that window. No flares -> empty list -> "no flare" label.

**`DatasetBuilder`** — runs the whole pipeline. Takes a file of image
timestamps and the flare catalog, walks through every image, applies the
matcher + strategy, and returns a `DataFrame` of `timestamp, label`.

## Input formats

### Image index CSV

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime string | Timestamp of the solar image (any pandas-parseable format, e.g. ISO 8601) |

A file with only the header row and no data rows is valid and produces an
empty result. A completely empty file (no header) raises `ValueError`, as
does a file missing the `timestamp` column.

### Flare catalog CSV

| Column | Type | Description |
|---|---|---|
| `date` | date string | Calendar date the flare occurred (kept for readability; not used for matching) |
| `start` | datetime string | When the flare began (not used for matching) |
| `peak` | datetime string | When the flare reached maximum intensity — **this is what matching uses** |
| `end` | datetime string | When the flare ended (not used for matching) |
| `class` | string | GOES class, e.g. `"M2.3"`, `"X1.0"` |
| `active_region` | string or empty | NOAA active region number, if known (not used for matching in v0.1.0 — see Future work) |

All six columns are required. Same empty-file rules as the image index:
header-only is valid and produces an empty catalog (nothing will ever
match); a completely empty file, or one missing any required column,
raises `ValueError` naming the problem and the file path.

## Window semantics

For an image taken at time `t` and a `prediction_window` of `N` hours,
`DatasetBuilder` looks for flares whose peak falls in the half-open
interval:

```
[t, t + N hours)
```

- Matching uses `peak_time` only — `start` and `end` are not used for the
  window check.
- The interval includes `t` itself and excludes the far endpoint exactly:
  a flare peaking at exactly `t + N` hours is **not** counted.
- There is no lead-time / gap in v0.1.0 — the window starts immediately at
  the image timestamp, with no buffer between the image and the
  prediction window. This is a deliberate choice for this release, not an
  oversight; a configurable gap is a possible future addition, not
  something silently missing.

## Labeling strategies

- **`BinaryThresholdStrategy(threshold="M")`** — label is `1` if any
  matched flare is at least the given GOES class, else `0`. Defaults to
  M-class.
- **`MaxFlareStrategy()`** — label is the numeric flux of the strongest
  matched flare, or `0.0` if none matched.

Both take the list of `FlareEvent`s returned by `EventMatcher.query()` and
reduce it to a single label; `DatasetBuilder` calls this once per image.

To use a different cutoff — for example, counting C-class and above as
positive instead of the M-class default:

```python
from solarflare_labeler import BinaryThresholdStrategy

strategy = BinaryThresholdStrategy(threshold="C")
```

## Scope for v0.1.0

This release supports exactly one labeling case:

- **Full-disk** — any flare in the time window counts, regardless of
  which part of the sun it came from.
- **Single-image** — one image timestamp produces one label; there is no
  grouping of consecutive images.
- **CSV inputs** — both the image index and the flare catalog are read
  from CSV files.

## Future work (not in v0.1.0)

These are deliberately out of scope for this release:

- **Active-region matching** — attributing a flare to the specific active
  region shown in an image, instead of counting any flare on the visible
  disk.
- **Sliding-window / series labeling** — grouping several consecutive
  images into one sample instead of labeling one image at a time.
- **Live catalog ingestion** — pulling flare data directly from NOAA /
  SolarSoft instead of reading a pre-made CSV.
- **A command-line interface** — today this is a library only; there is
  no CLI entry point.

## Development

```bash
pip install -e ".[dev]"
pytest
```
