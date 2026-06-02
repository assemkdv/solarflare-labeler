# solarflare-labeler — design notes

The plan is to rebuild this as OOP: the four labeling cases become their own classes, all sharing one base class. I'm reusing the terms from Balaji's HelioIndex (`cadence`, `observation_window`, `prediction_window`, `evals`, the `bl=` threshold) so the two packages stay consistent with each other.

This is still a draft — the class names, the exact arguments, and some of the single-vs-series logic are things I'd like to settle in the meeting.

## The four cases

It breaks down into a 2x2:

|                   | Single image        | Series of images          |
|-------------------|---------------------|---------------------------|
| **Full disk**     | `FullDiskSingle`    | `FullDiskSeries`          |
| **Active region** | `ActiveRegionSingle`| `ActiveRegionSeries`      |

Only two things actually differ across the four, which is why everything else can sit on a shared base:

- **Full disk vs active region** — how the matcher filters. Full disk takes any flare in the time window. Active region only counts flares from one specific region (by its NOAA number) and ignores everything else on the sun, even if a stronger flare went off elsewhere at the same time.
- **Single vs series** — how many images map to one label. Single is one image per label; series groups a sliding window of images and labels the group.

So the base class does the actual work, and each of the four overrides just the matching rule and the grouping rule.

## Base class: `FlareLabeler`

Holds everything the four cases share, so the state lives on an object instead of as module-level globals. (For what it's worth, HelioIndex keeps the catalog and timestamps as globals, which is exactly the thing that gets awkward once there are four different cases — so it's a decent argument for the OOP approach here.)

What it takes in:

- `flare_catalog` — the flare events
- `image_timestamps` — the images being labeled
- `observation_window` — how far back to look (the lookback)
- `prediction_window` — how far forward to scan for flares (the lookahead)
- `cadence` — how often to sample inside a window
- `threshold` — the cutoff for the binary label. Defaults to M, but not hardcoded — this is the point you raised last time
- `strategy` — how to reduce a set of flares to a single label (max / mean / flux)

What it does (shared, defined once):

- `classify()` — converts GOES class to flux on the log scale (A=1e-8 up to X=1e-4, times the multiplier). This is essentially my existing FlareClassifier.
- `apply_strategy()` — max / mean / flux
- `build()` — the main loop: walk through each image (or group), match, reduce to a label, return a row. Output is timestamp -> label.

What the four cases override:

- `match()` — full disk vs AR differ here
- `group()` — single vs series differ here

## The four classes

**`FullDiskSingle`** — time-window match, any flare counts, one image per label. This is basically what `DatasetBuilder` already does, so it's the only one that exists right now.

**`FullDiskSeries`** — same matching, but groups a sliding window of images and labels the group. Open question for the meeting: should the label come from the window after the *last* image in the group, or the *first*? I'm not sure which is right.

**`ActiveRegionSingle`** — matches by time and the region's NOAA number, ignoring stronger flares elsewhere. One image per label. Needs an extra `active_region` argument (the 5-digit number). The complication: a region is only labelable while it's actually on the visible disk, since regions rotate across it (~27 days for a full turn, east on the left and west on the right). So I need to decide what to output when the region isn't visible.

**`ActiveRegionSeries`** — AR matching plus the sliding-window grouping. The hardest case, since it's both at once.

## Things to raise at the meeting

1. **The separation gap — we seem to contradict each other.** Balaji added a `separation_minutes` gap between the image and the prediction window to prevent data leakage. You told me there shouldn't be a gap — the window starts right at the image timestamp. Which is correct, and should both packages handle it the same way?

2. **My matcher vs his.** His match is an exact date + start-time lookup, so it only catches an image if it lands exactly on a flare's start minute. Mine searches a time window (any flare peaking inside it), which I think is the right approach for labeling. I'd like to confirm I keep mine.

3. **Series label — last image or first?** (same as point 1 above)

4. **Active region off-disk** — what should I output when the target region isn't on the visible side of the sun?

5. **Which strategies are in scope?** The README has max and binary right now. You mentioned max / mean / average flux — I want to pin down which ones we're actually building.

## What I'm coding before the meeting (just this)

- Pull the hardcoded M out of `BinaryThresholdStrategy` and make it an argument, keeping M as the default.
- Everything else stays on paper until the design is approved, since the idea was to blueprint first.

## Housekeeping
- License: GPL 3.0
- Add `install_requires = pandas` to setup.cfg
