# Depth cost ([[Plan]]): anchored draws in training. Sites below self-decoder block K rerun only up to it and are
# scored against the error there (block K-1's mlp.c_proj error, from its own full draws, with the control variate).
# The recipe at the same draws (scale 1) and with twice the anchored sites' draws (scale 2, still cheaper than the
# recipe). 4 layers: K = 1 (the embedding and block 0's MLP input); 8 layers: K = 2 and 3. Baselines: the recipe in
# headline_tinystories_l4 / headline_copy_l4 and readout_l8. Seed 42.
from _common import COPY_TASK, L8, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42"] + LEAKFREE_CV
L4F = ["--local-scale", "1"]
L8F = ["--clip", "2"] + L8 + ["--local-scale", "4"]
TS = ["--data-dir", TINYSTORIES]
COPY = ["--data-dir", COPY_TASK, "--gamma", "1", "--cache-gamma", "0.98"]
OPTIONS = {".arm": {"values": ["l4_k1_x1", "l4_k1_x2", "l4copy_k1_x1", "l8_k2_x1", "l8_k2_x2", "l8_k3_x1", "l8_k3_x2"],
                    "flags": {"l4_k1_x1": L4F + TS + ["--anchor", "1"],
                              "l4_k1_x2": L4F + TS + ["--anchor", "1", "--anchor-scale", "2"],
                              "l4copy_k1_x1": L4F + COPY + ["--anchor", "1"],
                              "l8_k2_x1": L8F + TS + ["--anchor", "2"],
                              "l8_k2_x2": L8F + TS + ["--anchor", "2", "--anchor-scale", "2"],
                              "l8_k3_x1": L8F + TS + ["--anchor", "3"],
                              "l8_k3_x2": L8F + TS + ["--anchor", "3", "--anchor-scale", "2"]}}}
METRIC = "val_loss"
GOAL = "minimize"
