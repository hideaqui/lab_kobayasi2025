from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from main1022.training_base import run_training_pipeline  # type: ignore  # pylint: disable=import-error
else:
    from .training_base import run_training_pipeline


TOTAL_EPOCHS = 20
SEED = 1022
OUTPUT_TAG = "1022"


if __name__ == "__main__":
    run_training_pipeline(
        mode="normal",
        total_epoch=TOTAL_EPOCHS,
        seed=SEED,
        output_tag=OUTPUT_TAG,
    )
