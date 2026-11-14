# Cross-token readout at 8 layers, where the early-layer bias is larger: the full recipe with and without --readout,
# TinyStories (headline_tinystories_l8 settings) and the copy task (l8 settings from copy_extras_l4), 3000 steps,
# seed 42. Both arms use adaptive clipping (--clip 2), since the 8-layer copy runs spiked without it.
from _common import COPY_TASK, L8, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--clip", "2"] + L8 + LEAKFREE_CV + ["--local-scale", "4"]
OPTIONS = {
    ".data": {"values": ["tinystories", "copy"],
              "flags": {"tinystories": ["--data-dir", TINYSTORIES],
                        "copy": ["--data-dir", COPY_TASK, "--gamma", "1", "--cache-gamma", "0.98", "--lr", "0.15"]}},
    ".readout": {"values": [0, 1], "flags": {0: [], 1: ["--readout"]}},
    ".seed": {"values": [42], "flags": "--seed"},
}
METRIC = "val_loss"
GOAL = "minimize"
