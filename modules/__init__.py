"""
Real-Time Road Scene Analysis System — Module Package
=====================================================
Convenience re-exports so users can do:
    from modules import preprocessing, edge_detection, ...
"""

from modules import preprocessing
from modules import edge_detection
from modules import lane_detection
from modules import corner_detection
from modules import object_detector
from modules import object_tracker
from modules import classifier
from modules import metrics

__all__ = [
    "preprocessing",
    "edge_detection",
    "lane_detection",
    "corner_detection",
    "object_detector",
    "object_tracker",
    "classifier",
    "metrics",
]
