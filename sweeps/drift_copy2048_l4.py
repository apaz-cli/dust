# Drift test, harder version: start from the backprop model trained to copy from 1024 tokens back (T = 2048,
# ckpt/bp_copy2048_l4 <- copy_length_l4_20261114_1817_ef37 backprop, copied half 0.069) and keep training on the same
# task. Noise-free Dust with gamma 0.98 has essentially no credit at that distance (0.98^1024), so this is the case
# where Dust's bias should matter most. Also the recipe at a 5x lower learning rate, gentler on a trained model.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tsrepeat2048", "--batch-seqs", "2",
           "--init", "ckpt/bp_copy2048_l4", "--halves-every"]
RECIPE = LEAKFREE_CV + ["--local-scale", "1", "--gamma", "1", "--cache-gamma", "0.98"]
OPTIONS = {".arm": {"values": ["backprop", "noisefree_g98", "noisefree_g1", "recipe", "recipe_lr05"],
                    "flags": {"backprop": ["--method", "bp", "--lr", "0.2"],
                              "noisefree_g98": ["--method", "ed", "--gamma", "0.98", "--lr", "0.2"],
                              "noisefree_g1": ["--method", "ed", "--gamma", "1", "--lr", "0.2"],
                              "recipe": RECIPE,
                              "recipe_lr05": RECIPE + ["--lr", "0.05"]}}}
METRIC = "val_second_half"
GOAL = "minimize"
