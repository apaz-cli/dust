# The readout plus twice the residual-stream draws at 8 layers (at 4 layers they add up, readout_combine). TinyStories
# and the copy task, seed 42, readout with a 100-step warm-up. Compare readout_combine (twice the draws alone) and
# readout_l8 / readout_l8_warmup (readout alone).
from _common import COPY_TASK, L8, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--clip", "2", "--local-scale", "4",
           "--direct-scale", "2", "--emb-scale", "8", "--readout", "--readout-warmup", "100"] + L8 + LEAKFREE_CV
OPTIONS = {".data": {"values": ["tinystories", "copy"],
                     "flags": {"tinystories": ["--data-dir", TINYSTORIES],
                               "copy": ["--data-dir", COPY_TASK, "--gamma", "1", "--cache-gamma", "0.98", "--lr", "0.15"]}}}
METRIC = "val_loss"
GOAL = "minimize"
