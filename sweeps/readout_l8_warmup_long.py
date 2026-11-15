# The 300-step warm-up gave a much larger 8-layer gain than 100 steps (readout_l8_warmup, seed 42). Longer warm-ups
# (600, 1000) on seed 42, 300 steps on seeds 7 and 1 (their plain runs: readout_l8_warmup seed 7, readout_l8_seeds
# seed 1), and 300 steps on the copy task. Settings as readout_l8_seeds.
from _common import COPY_TASK, L8, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

TS = ["--data-dir", TINYSTORIES]
COPY = ["--data-dir", COPY_TASK, "--gamma", "1", "--cache-gamma", "0.98", "--lr", "0.15"]
COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--clip", "2", "--readout"] + L8 + LEAKFREE_CV + ["--local-scale", "4"]
OPTIONS = {".arm": {"values": ["ts_w600", "ts_w1000", "ts_w300_seed7", "ts_w300_seed1", "copy_w300"],
                    "flags": {"ts_w600": TS + ["--readout-warmup", "600", "--seed", "42"],
                              "ts_w1000": TS + ["--readout-warmup", "1000", "--seed", "42"],
                              "ts_w300_seed7": TS + ["--readout-warmup", "300", "--seed", "7"],
                              "ts_w300_seed1": TS + ["--readout-warmup", "300", "--seed", "1"],
                              "copy_w300": COPY + ["--readout-warmup", "300", "--seed", "42"]}}}
METRIC = "val_loss"
GOAL = "minimize"
