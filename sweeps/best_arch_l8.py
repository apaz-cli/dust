# The candidate Dust-friendly architecture at 8 layers, real Dust: a 2-layer self-decoder (backprop loses almost
# nothing there, self_depth_bp) with the recipe plus the cross-token readout (100-step warm-up). TinyStories and the
# copy task, seed 42. Compare self_depth_dust (same without the readout) and readout_l8 (default 4-layer self-decoder).
from _common import COPY_TASK, L8, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--clip", "2", "--local-scale", "4", "--n-self", "2",
           "--readout", "--readout-warmup", "100"] + L8 + LEAKFREE_CV
OPTIONS = {".data": {"values": ["tinystories", "copy"],
                     "flags": {"tinystories": ["--data-dir", TINYSTORIES],
                               "copy": ["--data-dir", COPY_TASK, "--gamma", "1", "--cache-gamma", "0.98", "--lr", "0.15"]}}}
METRIC = "val_loss"
GOAL = "minimize"
