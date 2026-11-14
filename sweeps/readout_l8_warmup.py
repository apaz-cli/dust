# The 8-layer TinyStories run with --readout diverged at step 33 (readout_l8). Is it the early noisy targets
# (fix: plain Dust for the first steps, --readout-warmup) or the seed? Same settings as readout_l8.
from _common import L8, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--clip", "2", "--data-dir", TINYSTORIES] + L8 + LEAKFREE_CV + [
    "--local-scale", "4"]
OPTIONS = {".arm": {"values": ["warmup100", "warmup300", "nowarmup_seed7", "plain_seed7"],
                    "flags": {"warmup100": ["--readout", "--readout-warmup", "100", "--seed", "42"],
                              "warmup300": ["--readout", "--readout-warmup", "300", "--seed", "42"],
                              "nowarmup_seed7": ["--readout", "--seed", "7"],
                              "plain_seed7": ["--seed", "7"]}}}
METRIC = "val_loss"
GOAL = "minimize"
