class FlareClassifier:
    """
    Converts GOES flare class letters to numeric X-ray flux values (W/m²).

    Scale (logarithmic):
        A → 1e-8
        B → 1e-7
        C → 1e-6
        M → 1e-5
        X → 1e-4
    """

    FLUX_MAP = {
        "A": 1e-8,
        "B": 1e-7,
        "C": 1e-6,
        "M": 1e-5,
        "X": 1e-4,
    }

    def to_flux(self, goes_class: str) -> float:
        """
        Convert a GOES class string to numeric flux.

        Examples: "M2.3" → 2.3e-5,  "C4" → 4e-6
        """
        letter = goes_class[0].upper()
        number = float(goes_class[1:]) if len(goes_class) > 1 else 1.0
        base = self.FLUX_MAP[letter]
        return base * number

    def is_strong(self, goes_class: str, threshold: str = "M") -> bool:
        """
        Returns True if the flare meets or exceeds the threshold class.

        Example: is_strong("M2.3") → True, is_strong("C4") → False
        """
        return self.to_flux(goes_class) >= self.FLUX_MAP[threshold]