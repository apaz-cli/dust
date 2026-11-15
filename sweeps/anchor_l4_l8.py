# Depth cost ([[Plan]]): anchored draws in training. Sites below self-decoder block K rerun only up to it and are
# scored against the error there (block K-1's mlp.c_proj error, from its own full draws, with the control variate).
# The recipe at the same draws (scale 1) and with twice the anchored sites' draws (scale 2, still cheaper than the
# recipe). 8 layers, K = 2 and 3 (the embedding keeps full draws: it also feeds every block directly; at 4 layers
# that leaves only block 0's MLP input below the one possible anchor, so 4 layers is not run). Baselines: the recipe in
# readout_l8. Seed 42.
from _common import COPY_TASK, L8, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42"] + LEAKFREE_CV
L8F = ["--clip", "2"] + L8 + ["--local-scale", "4"]
TS = ["--data-dir", TINYSTORIES]
OPTIONS = {".arm": {"values": ["l8_k2_x1", "l8_k2_x2", "l8_k3_x1", "l8_k3_x2"],
                    "flags": {"l8_k2_x1": L8F + TS + ["--anchor", "2"],
                              "l8_k2_x2": L8F + TS + ["--anchor", "2", "--anchor-scale", "2"],
                              "l8_k3_x1": L8F + TS + ["--anchor", "3"],
                              "l8_k3_x2": L8F + TS + ["--anchor", "3", "--anchor-scale", "2"]}}}
METRIC = "val_loss"
GOAL = "minimize"
