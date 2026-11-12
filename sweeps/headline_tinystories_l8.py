# Headline table, TinyStories, 8-layer YOCO (32M params), 3000 steps x 8 sequences x 512 tokens, 1 seed.
# At 8 layers the leak-free arms keep 4x local draws (1x was not enough there).
from _common import BEST, L8, LEAKFREE, LEAKFREE_CV, STOCK, TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

COMMAND = command(TINYSTORIES) + L8
OPTIONS = {
    ".arm": {
        "values": ["backprop", "noisefree", "stock", "best", "leakfree", "leakfree_cv"],
        "flags": {
            "backprop": ["--method", "bp", "--lr", "0.1"],
            "noisefree": ["--method", "ed", "--gamma", "0.98", "--lr", "0.1"],
            "stock": STOCK,
            "best": BEST + ["--local-scale", "4"],
            "leakfree": LEAKFREE + ["--local-scale", "4"],
            "leakfree_cv": LEAKFREE_CV + ["--local-scale", "4"],
        },
    },
    ".seed": {"values": [42], "flags": "--seed"},
}
METRIC = "val_loss"
GOAL = "minimize"
