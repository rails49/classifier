"""The rails49 occupancy classifier: cut patches from a frame and score them. The
training code needs the ``train`` extra."""

from .runtime import MODEL, Classifier, Result, Sensor

__all__ = ["MODEL", "Classifier", "Result", "Sensor"]
