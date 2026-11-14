# More seeds for readout_l4 (which had 42 and 7): 4-layer YOCO, 3000 steps, recipe with and without --readout.
from _common import COPY_TASK, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco"] + LEAKFREE_CV + ["--local-scale", "1"]
OPTIONS = {
    ".data": {"values": ["tinystories", "copy"],
              "flags": {"tinystories": ["--data-dir", TINYSTORIES],
                        "copy": ["--data-dir", COPY_TASK, "--gamma", "1", "--cache-gamma", "0.98"]}},
    ".readout": {"values": [0, 1], "flags": {0: [], 1: ["--readout"]}},
    ".seed": {"values": [1, 2, 3], "flags": "--seed"},
}
METRIC = "val_loss"
GOAL = "minimize"
