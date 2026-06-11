from .classifier import FlareClassifier
from .events import FlareEvent

_classifier = FlareClassifier()


class BinaryThresholdStrategy:
    """Returns 1 if any flare in the window is M-class or stronger, else 0."""

    def label(self, flares: list[FlareEvent]) -> int:
        return int(any(_classifier.is_strong(f.goes_class) for f in flares))


class MaxFlareStrategy:
    """Returns the numeric flux of the strongest flare in the window, or 0.0."""

    def label(self, flares: list[FlareEvent]) -> float:
        if not flares:
            return 0.0
        return max(_classifier.to_flux(f.goes_class) for f in flares)
