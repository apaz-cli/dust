# Learning-rate sweep for backprop, dense vs MoE, SGD and AdamW, 4-layer TinyStories, 1500 steps, seed 42.
# The expert lr multiplier (moe_m4) applies to SGD only (it rebuilds the optimizer as SGD).
from _common import TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

COMMAND = command(TINYSTORIES) + ["--method", "bp", "--steps", "1500"]
OPTIONS = {
    ".opt": {
        ".sgd": {".lr": {"values": [0.05, 0.1, 0.2, 0.4], "flags": "--lr"}},
        ".adamw": {"flags": ["--opt", "adamw"],
                  ".adam_lr": {"values": [3e-4, 1e-3, 3e-3, 1e-2], "flags": "--adam-lr"}},
    },
    ".mlp": {"values": ["dense", "moe", "moe_m4"],
             "flags": {"dense": [], "moe": ["--moe", "8,2,384,256"],
                       "moe_m4": ["--moe", "8,2,384,256", "--moe-lr-mult", "4"]}},
    ".seed": {"values": [42], "flags": "--seed"},
}


def EXCLUDE(combo):
    return combo.get("opt") == "adamw" and combo.get("mlp") == "moe_m4"


METRIC = "val_loss"
GOAL = "minimize"
