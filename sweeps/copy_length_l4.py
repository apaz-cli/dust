# Sequence-length ladder on the copy task: the copied half is T/2 tokens back, T = 2048 and 8192 (the T = 512 point
# is headline_copy_l4). Tokens per step held at 4096 where possible (8 x 512, 2 x 2048); T = 8192 is 1 sequence,
# so 8192 tokens per step. Backprop and noise-free Dust as references, the recipe (leak-free scoring, live units,
# control variate) with the long credit horizon, and the recipe plus the cross-token readout at T = 2048.
# 4 layers, 3000 steps, seed 42. Data: yoco/make_repeat_long.py.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

LONG = ["--gamma", "1", "--cache-gamma", "0.98", "--local-scale", "1"]
COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42"]
OPTIONS = {
    ".T": {"values": [2048, 8192],
           "flags": {2048: ["--data-dir", "data/tsrepeat2048", "--batch-seqs", "2"],
                     8192: ["--data-dir", "data/tsrepeat8192", "--batch-seqs", "1"]}},
    ".arm": {"values": ["backprop", "noisefree", "leakfree_cv", "leakfree_cv_readout"],
             "flags": {"backprop": ["--method", "bp", "--lr", "0.2"],
                       "noisefree": ["--method", "ed", "--gamma", "1", "--lr", "0.2"],
                       "leakfree_cv": LEAKFREE_CV + LONG,
                       "leakfree_cv_readout": LEAKFREE_CV + LONG + ["--readout"]}},
}


def EXCLUDE(combo):  # the readout's pairwise tensors are too big at 8192
    return combo.get("T") == 8192 and combo.get("arm") == "leakfree_cv_readout"


METRIC = "val_second_half"
GOAL = "minimize"
