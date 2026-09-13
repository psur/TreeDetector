"""Registered, implemented benchmark backends."""
from .yolo_model import YoloModel

_BACKENDS = {"yolo": YoloModel}

def get_model(name, cfg, experiment=None):
    if name not in _BACKENDS:
        raise ValueError(f"Unknown model: {name}")
    return _BACKENDS[name](cfg, experiment)

def model_names():
    return tuple(_BACKENDS)
