# The readout at 8 layers with the warm-up that stopped its early divergence (100 steps of plain Dust first,
# readout_l8_warmup), on more seeds and on the copy task, with matching runs without it. A shorter warm-up (25, 50)
# on seed 42 checks how much is needed. Settings as readout_l8.
from _common import COPY_TASK, L8, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

TS = ["--data-dir", TINYSTORIES]
COPY = ["--data-dir", COPY_TASK, "--gamma", "1", "--cache-gamma", "0.98", "--lr", "0.15"]
RO = ["--readout", "--readout-warmup", "100"]
COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--clip", "2"] + L8 + LEAKFREE_CV + ["--local-scale", "4"]
OPTIONS = {".arm": {"values": ["ts_ro_seed7", "ts_ro_seed1", "ts_plain_seed1", "copy_ro_seed7", "copy_plain_seed7",
                               "ts_ro_warmup25", "ts_ro_warmup50"],
                    "flags": {"ts_ro_seed7": TS + RO + ["--seed", "7"],
                              "ts_ro_seed1": TS + RO + ["--seed", "1"],
                              "ts_plain_seed1": TS + ["--seed", "1"],
                              "copy_ro_seed7": COPY + RO + ["--seed", "7"],
                              "copy_plain_seed7": COPY + ["--seed", "7"],
                              "ts_ro_warmup25": TS + ["--readout", "--readout-warmup", "25", "--seed", "42"],
                              "ts_ro_warmup50": TS + ["--readout", "--readout-warmup", "50", "--seed", "42"]}}}
METRIC = "val_loss"
GOAL = "minimize"
