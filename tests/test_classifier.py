from solarflare_labeler.classifier import FlareClassifier

classifier = FlareClassifier()

# Test to_flux
assert classifier.to_flux("M2.3") == 2.3e-5
assert classifier.to_flux("C4.0") == 4e-6
assert classifier.to_flux("X1.0") == 1e-4
assert classifier.to_flux("A1.0") == 1e-8

# Test is_strong
assert classifier.is_strong("M2.3") == True
assert classifier.is_strong("X1.0") == True
assert classifier.is_strong("C4.0") == False
assert classifier.is_strong("B1.0") == False

print("all tests passed")