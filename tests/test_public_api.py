import solarflare_labeler as sfl


def test_public_names_are_accessible():
    expected = {
        "FlareClassifier",
        "FlareEvent",
        "EventMatcher",
        "BinaryThresholdStrategy",
        "MaxFlareStrategy",
        "DatasetBuilder",
    }

    assert set(sfl.__all__) == expected
    for name in expected:
        assert hasattr(sfl, name)


def test_top_level_classes_are_usable():
    strategy = sfl.BinaryThresholdStrategy()
    assert strategy.threshold == "M"

    builder = sfl.DatasetBuilder(prediction_window=24, strategy=sfl.BinaryThresholdStrategy())
    assert builder.prediction_window == 24
