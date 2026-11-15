# Plan item 2, small version of the drift test: does Dust training make a model forget how to use context? Start from
# the backprop-trained copy-task model (copies nearly perfectly: copied half 0.049) and keep training it on the copy
# task for 3000 steps with (a) backprop, (b) noise-free Dust with gamma 0.98 (Dust's bias, no noise), (c) noise-free
# Dust with gamma 1, (d) the recipe with gamma 1 in the self-decoder, (e) stock Dust. Both halves' loss logged at every
# evaluation (--halves-every), so the copied half's drift is visible over time. 4 layers, seed 42.
from _common import BP_COPY, COPY_TASK, LEAKFREE_CV, PY, STOCK, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", COPY_TASK, "--init", BP_COPY,
           "--halves-every"]
OPTIONS = {".arm": {"values": ["backprop", "noisefree_g98", "noisefree_g1", "recipe", "stock"],
                    "flags": {"backprop": ["--method", "bp", "--lr", "0.2"],
                              "noisefree_g98": ["--method", "ed", "--gamma", "0.98", "--lr", "0.2"],
                              "noisefree_g1": ["--method", "ed", "--gamma", "1", "--lr", "0.2"],
                              "recipe": LEAKFREE_CV + ["--local-scale", "1", "--gamma", "1", "--cache-gamma", "0.98"],
                              "stock": STOCK}}}
METRIC = "val_second_half"
GOAL = "minimize"
