# Rerun of the backprop arms at 64 sequences from moe_batch_l4_20261115_1002_5ed3, which crashed in attention (SDPA with a
# mask returned None at that batch size on the B300s; model.attention now runs large batches in chunks of 16).
from moe_batch_l4 import COMMAND, GOAL, METRIC, OPTIONS, UPLOAD_EXCLUDE  # noqa: F401


def EXCLUDE(combo):
    return not (combo["method"].startswith("bp") and combo["seqs"] == 64)
