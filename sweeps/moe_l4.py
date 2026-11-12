# Mixture of experts vs dense, 4-layer TinyStories, 1500 steps, seed 42 (MoE under Dust note).
# MoE: 8 experts, top 2, hidden 384 each, plus a shared expert of 256 (active hidden 1024 = dense).
# wide: a dense MLP as wide as the MoE's total hidden (3072 + 256).
from _common import BEST, TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

COMMAND = command(TINYSTORIES) + ["--steps", "1500"]
OPTIONS = {
    ".method": {
        "values": ["bp_sgd", "bp_adamw", "dust", "dust_adamw"],
        "flags": {
            "bp_sgd": ["--method", "bp", "--lr", "0.2"],
            "bp_adamw": ["--method", "bp", "--opt", "adamw", "--adam-lr", "3e-3"],
            "dust": BEST + ["--local-scale", "4"],
            "dust_adamw": BEST + ["--local-scale", "4", "--opt", "adamw", "--adam-lr", "1e-3"],
        },
    },
    ".mlp": {
        "values": ["dense", "moe", "wide"],
        "flags": {"dense": [], "moe": ["--moe", "8,2,384,256"], "wide": ["--moe", "1,1,3072,256"]},
    },
}
METRIC = "val_loss"
GOAL = "minimize"
