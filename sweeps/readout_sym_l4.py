# Plan item 2: the symmetric readout in training. 4 layers, 3000 steps, the recipe with --readout-mode leak (only
# cancel the predicted leakage into each token's own reward) and both (that plus the readout's credit). The credit
# mode alone is readout_l4. TinyStories and the copy task, seed 42.
from _common import COPY_TASK, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--readout", "--local-scale", "1", "--seed", "42"] + LEAKFREE_CV
OPTIONS = {
    ".data": {"values": ["tinystories", "copy"],
              "flags": {"tinystories": ["--data-dir", TINYSTORIES],
                        "copy": ["--data-dir", COPY_TASK, "--gamma", "1", "--cache-gamma", "0.98"]}},
    ".mode": {"values": ["leak", "both"], "flags": "--readout-mode"},
}
METRIC = "val_loss"
GOAL = "minimize"
