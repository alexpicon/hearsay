# Author: Alex Picon <alexnpc@me.com>
"""Public entry point: ``from hearsay.detector import score_file``."""

from hearsay.inference import ScoreResult, get_detector, score_directory, score_file

__all__ = ["ScoreResult", "get_detector", "score_directory", "score_file"]
