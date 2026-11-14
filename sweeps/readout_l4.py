# Cross-token readout in training (--readout: blocks top-down, each draw also scores the closed-form change of later
# tokens' attention outputs from its own keys and values, against the chained attention-output targets), against the
# same recipe without it. 4-layer YOCO, 3000 steps, 2 seeds; TinyStories and the copy task (headline settings).
from _common import COPY_TASK, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

LONG = ["--gamma", "1", "--cache-gamma", "0.98"]
COMMAND = [PY, "yoco/train.py", "--arch", "yoco"] + LEAKFREE_CV + ["--local-scale", "1"]
OPTIONS = {
    ".data": {"values": ["tinystories", "copy"],
              "flags": {"tinystories": ["--data-dir", TINYSTORIES], "copy": ["--data-dir", COPY_TASK] + LONG}},
    ".readout": {"values": [0, 1], "flags": {0: [], 1: ["--readout"]}},
    ".seed": {"values": [42, 7], "flags": "--seed"},
}
METRIC = "val_loss"
GOAL = "minimize"
