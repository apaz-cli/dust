# Headline table, TinyStories, 4-layer YOCO (6.2M params), 3000 steps x 8 sequences x 512 tokens, 2 seeds.
# Reruns of results first logged before mlsweep (see sources/DustYOCO/FINDINGS.md).
from _common import BEST, LEAKFREE, LEAKFREE_CV, STOCK, TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

COMMAND = command(TINYSTORIES)
OPTIONS = {
    ".arm": {
        "values": ["backprop", "noisefree", "stock", "best", "leakfree", "leakfree_cv"],
        "flags": {
            "backprop": ["--method", "bp", "--lr", "0.2"],
            "noisefree": ["--method", "ed", "--gamma", "0.98", "--lr", "0.2"],
            "stock": STOCK,
            "best": BEST + ["--local-scale", "4"],
            "leakfree": LEAKFREE + ["--local-scale", "1"],
            "leakfree_cv": LEAKFREE_CV + ["--local-scale", "1"],
        },
    },
    ".seed": {"values": [42, 7], "flags": "--seed"},
}
METRIC = "val_loss"
GOAL = "minimize"
