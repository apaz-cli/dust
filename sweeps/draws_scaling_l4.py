# How does the gain grow with draws? The recipe with 4x the residual-stream draws, with and without the readout
# (2x is readout_compute_l4 / readout_combine). TinyStories and the copy task, 4 layers, seed 42.
from _common import COPY_TASK, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--local-scale", "1", "--direct-scale", "4",
           "--emb-scale", "16"] + LEAKFREE_CV
OPTIONS = {
    ".data": {"values": ["tinystories", "copy"],
              "flags": {"tinystories": ["--data-dir", TINYSTORIES],
                        "copy": ["--data-dir", COPY_TASK, "--gamma", "1", "--cache-gamma", "0.98"]}},
    ".readout": {"values": [0, 1], "flags": {0: [], 1: ["--readout"]}},
}
METRIC = "val_loss"
GOAL = "minimize"
