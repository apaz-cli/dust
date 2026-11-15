# Depth cost ([[Plan]]): anchored draws in training. Sites below self-decoder block K rerun only up to it and are
# scored against the error there (block K-1's mlp.c_proj error, from its own full draws, with the control variate).
# The anchor's error is shared by every anchored draw, so its noise does not average over their draws (the first
# version, without these, diverged: the anchored estimates were 4x larger, mostly the anchor's noise). So the
# anchor's writers get B times their draws (boost), and anchoring starts after 200 steps of ordinary draws.
# 8 layers (the embedding keeps full draws: it also feeds every block directly). Baseline: the recipe in readout_l8.
from _common import L8, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", TINYSTORIES, "--clip", "2",
           "--local-scale", "4", "--anchor-warmup", "200"] + L8 + LEAKFREE_CV
OPTIONS = {".arm": {"values": ["k3_b4", "k3_b8", "k2_b8", "k3_b8_x2"],
                    "flags": {"k3_b4": ["--anchor", "3", "--anchor-boost", "4"],
                              "k3_b8": ["--anchor", "3", "--anchor-boost", "8"],
                              "k2_b8": ["--anchor", "2", "--anchor-boost", "8"],
                              "k3_b8_x2": ["--anchor", "3", "--anchor-boost", "8", "--anchor-scale", "2"]}}}
METRIC = "val_loss"
GOAL = "minimize"
