# Rerun of the 64-sequence arms of moe_batch_l4_20261115_1002_5ed3: 64 x 1500 sequences is more than the 72k rows, and
# the data loader did not wrap around, so step 1126 got an empty batch and crashed (attention returned None). Fixed in
# yoco/train.py (the loader now starts another pass).
from moe_batch_l4 import COMMAND, GOAL, METRIC, OPTIONS, UPLOAD_EXCLUDE  # noqa: F401


def EXCLUDE(combo):
    return combo["seqs"] != 64
