# Plan item 1, the Dust side: does Dust's gap to backprop shrink with a shallow self-decoder? The recipe at
# 4 layers with a 1-layer self-decoder (the default 2 is headline_tinystories_l4 / readout_l4), and at 8 layers with
# 1 and 2 (the default 4 is readout_l8). TinyStories and the copy task (gamma 1 in the self-decoder, 0.98 on the
# cache; lr 0.15 at 8 layers as there). Compare self_depth_bp.
from _common import COPY_TASK, L8, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco"] + LEAKFREE_CV
L4F = ["--local-scale", "1"]
L8F = ["--clip", "2"] + L8 + ["--local-scale", "4"]
TS = ["--data-dir", TINYSTORIES]
COPY = ["--data-dir", COPY_TASK, "--gamma", "1", "--cache-gamma", "0.98"]
OPTIONS = {".arm": {"values": ["l4_n1_ts_s42", "l4_n1_ts_s7", "l4_n1_copy_s42", "l4_n1_copy_s7",
                               "l8_n1_ts", "l8_n2_ts", "l8_n1_copy", "l8_n2_copy"],
                    "flags": {"l4_n1_ts_s42": L4F + TS + ["--n-self", "1", "--seed", "42"],
                              "l4_n1_ts_s7": L4F + TS + ["--n-self", "1", "--seed", "7"],
                              "l4_n1_copy_s42": L4F + COPY + ["--n-self", "1", "--seed", "42"],
                              "l4_n1_copy_s7": L4F + COPY + ["--n-self", "1", "--seed", "7"],
                              "l8_n1_ts": L8F + TS + ["--n-self", "1", "--seed", "42"],
                              "l8_n2_ts": L8F + TS + ["--n-self", "2", "--seed", "42"],
                              "l8_n1_copy": L8F + COPY + ["--lr", "0.15", "--n-self", "1", "--seed", "42"],
                              "l8_n2_copy": L8F + COPY + ["--lr", "0.15", "--n-self", "2", "--seed", "42"]}}}
METRIC = "val_loss"
GOAL = "minimize"
