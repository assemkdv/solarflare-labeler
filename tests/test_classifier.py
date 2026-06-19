import pandas as pd
import pytest
from solarflare_labeler.classifier import FlareClassifier
from solarflare_labeler.events import EventMatcher, FlareEvent

# ── FlareClassifier tests ──────────────────────────────────────────────


def test_to_flux():
    classifier = FlareClassifier()
    assert classifier.to_flux("M2.3") == 2.3e-5
    assert classifier.to_flux("C4.0") == 4e-6
    assert classifier.to_flux("X1.0") == 1e-4
    assert classifier.to_flux("A1.0") == 1e-8


def test_is_strong():
    classifier = FlareClassifier()
    assert classifier.is_strong("M2.3") == True
    assert classifier.is_strong("X1.0") == True
    assert classifier.is_strong("C4.0") == False
    assert classifier.is_strong("B1.0") == False


def test_to_flux_invalid_letter_raises():
    classifier = FlareClassifier()
    with pytest.raises(KeyError):
        classifier.to_flux("Z5.0")


def test_to_flux_empty_string_raises():
    classifier = FlareClassifier()
    with pytest.raises(IndexError):
        classifier.to_flux("")


def test_to_flux_non_numeric_suffix_raises():
    classifier = FlareClassifier()
    with pytest.raises(ValueError):
        classifier.to_flux("M2.3.4")


# ── EventMatcher tests ─────────────────────────────────────────────────


def test_query_returns_flares_within_window():
    # Build a tiny fake catalog instead of loading a real CSV
    # This lets us test the logic without needing the NOAA dataset yet
    matcher = EventMatcher.__new__(EventMatcher)  # create object without calling __init__
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime([
            "2024-01-01 00:30",  # inside a 1-hour window starting at 00:00
            "2024-01-01 01:30",  # outside — too late
            "2024-01-01 00:59",  # inside — right at the edge
        ]),
        "start_time": pd.to_datetime([
            "2024-01-01 00:20",
            "2024-01-01 01:20",
            "2024-01-01 00:50",
        ]),
        "goes_class": ["M2.3", "X1.0", "C4.5"],
        "active_region": [None, None, None],
    }).sort_values("peak_time").reset_index(drop=True)

    # Query with a 1-hour prediction window starting at midnight
    results = matcher.query(
        image_time=pd.Timestamp("2024-01-01 00:00"),
        prediction_window_hours=1,
    )

    # Should find 2 flares (M2.3 and C4.5) — X1.0 is outside the window
    assert len(results) == 2
    assert results[0].goes_class == "M2.3"
    assert results[1].goes_class == "C4.5"
