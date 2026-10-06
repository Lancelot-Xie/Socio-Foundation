"""train_ppo entry with a transformers>=5 compat shim for verl 0.7.

transformers 5.x removed `AutoModelForVision2Seq` (renamed to
`AutoModelForImageTextToText`); verl 0.7 still imports the old name at module
load. We alias it back before importing verl so the (text-only) RL path works
with a transformers new enough to recognize the `qwen3_5` architecture.
"""
import os
import sys

# scripts/ is this file's dir; verl lives in the repo root one level up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transformers

if not hasattr(transformers, "AutoModelForVision2Seq") and hasattr(
    transformers, "AutoModelForImageTextToText"
):
    transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText

from verl.trainer.main_ppo import main  # noqa: E402

if __name__ == "__main__":
    main()
