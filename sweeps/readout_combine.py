# The readout (removes part of the bias) tied or beat twice the residual draws (cuts noise) at 4 layers
# (readout_compute_l4). Do the two add? 4 layers: readout plus twice the draws, TinyStories and the copy task,
# seeds 42 and 7. 8 layers: the equal-compute control (twice the draws, no readout), TinyStories seeds 42 and 7.
from _common import COPY_TASK, L8, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

X2 = ["--direct-scale", "2", "--emb-scale", "8"]
TS = ["--data-dir", TINYSTORIES]
COPY = ["--data-dir", COPY_TASK, "--gamma", "1", "--cache-gamma", "0.98"]
L4RO = ["--local-scale", "1", "--readout"] + X2
L8X2 = ["--clip", "2"] + L8 + ["--local-scale", "4"] + X2
COMMAND = [PY, "yoco/train.py", "--arch", "yoco"] + LEAKFREE_CV
OPTIONS = {".arm": {"values": ["l4_ts_seed42", "l4_ts_seed7", "l4_copy_seed42", "l4_copy_seed7",
                               "l8_x2_seed42", "l8_x2_seed7"],
                    "flags": {"l4_ts_seed42": L4RO + TS + ["--seed", "42"],
                              "l4_ts_seed7": L4RO + TS + ["--seed", "7"],
                              "l4_copy_seed42": L4RO + COPY + ["--seed", "42"],
                              "l4_copy_seed7": L4RO + COPY + ["--seed", "7"],
                              "l8_x2_seed42": L8X2 + TS + ["--seed", "42"],
                              "l8_x2_seed7": L8X2 + TS + ["--seed", "7"]}}}
METRIC = "val_loss"
GOAL = "minimize"
