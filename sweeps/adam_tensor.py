# AdamW gets less than SGD out of Dust's noise reductions, mostly the control variate (adam_dust_cause). Adam with one
# second moment per tensor (--opt adam_tensor) keeps the update's direction within each tensor. Real Dust, the recipe,
# 1500 steps, TinyStories, 4 layers, dense and MoE (8 experts, 2 active, shared expert: moe_l4's setting).
# Compare SGD 2.556 / AdamW 2.643 dense, SGD 2.817 / AdamW 3.129 MoE.
from _common import LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", TINYSTORIES, "--steps", "1500",
           "--local-scale", "1", "--opt", "adam_tensor"] + LEAKFREE_CV
OPTIONS = {
    ".mlp": {"values": ["dense", "moe"], "flags": {"dense": [], "moe": ["--moe", "8,2,384,256"]}},
    ".lr": {"values": [1e-3, 3e-3, 1e-2], "flags": "--adam-lr"},
}
METRIC = "val_loss"
GOAL = "minimize"
