# The dense references of moe_next_round, rerun: on the B300s (torch 2.11) the compiled model forward gave NaN at the
# first step (wrong per-token losses on reruns from block 0). The training script now checks compiled against eager at
# startup and falls back to the eager forward when they disagree (the head draws stay compiled).
from moe_next_round import COMMAND, GOAL, METRIC, OPTIONS, UPLOAD_EXCLUDE  # noqa: F401


def EXCLUDE(combo):
    return combo["arm"] not in ("dense_s64", "dense_s128", "dense_s64_6k")
