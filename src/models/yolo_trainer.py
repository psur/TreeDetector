"""Ultralytics trainer extensions for reliable checkpoint writes on Windows."""
from __future__ import annotations

import logging
import time

from ultralytics.models.yolo.segment import SegmentationTrainer

LOG = logging.getLogger(__name__)


class RetryingSegmentationTrainer(SegmentationTrainer):
    """Retry transient Windows checkpoint locks without changing checkpoint content."""

    checkpoint_write_attempts = 8

    def save_model(self):
        for attempt in range(1, self.checkpoint_write_attempts + 1):
            try:
                return super().save_model()
            except PermissionError:
                if attempt == self.checkpoint_write_attempts:
                    raise
                delay = min(0.5 * (2 ** (attempt - 1)), 8.0)
                LOG.warning(
                    "Checkpoint file is temporarily locked; retrying save in %.1f seconds (%d/%d)",
                    delay,
                    attempt,
                    self.checkpoint_write_attempts,
                )
                time.sleep(delay)
