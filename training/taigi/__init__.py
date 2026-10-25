"""Prepare weakly labeled Chinese-semantic to Taigi speech data."""

from training.taigi.pipeline import prepare_candidates, transcribe_release

__all__ = ["prepare_candidates", "transcribe_release"]
