# Extra copy-task rows for the Context use note (the main ones are in headline_copy_l4). 4 layers, seed 42.
from _common import COPY_TASK, LEAKFREE, LEAKFREE_CV, L8, UPLOAD_EXCLUDE, command  # noqa: F401

COMMAND = command(COPY_TASK)
OPTIONS = {
    ".arm": {
        "values": ["noisefree_g98", "leakfree_g98", "leakfree_g1_cache_only", "leakfree_g1_everywhere",
                   "l8_backprop", "l8_noisefree", "l8_leakfree_lr15", "l8_leakfree_cv_lr15"],
        "flags": {
            "noisefree_g98": ["--method", "ed", "--gamma", "0.98", "--lr", "0.2"],
            "leakfree_g98": LEAKFREE + ["--local-scale", "1"],
            "leakfree_g1_cache_only": LEAKFREE + ["--local-scale", "1", "--cache-gamma", "1"],
            "leakfree_g1_everywhere": LEAKFREE + ["--local-scale", "1", "--gamma", "1"],
            "l8_backprop": L8 + ["--method", "bp", "--lr", "0.1"],
            "l8_noisefree": L8 + ["--method", "ed", "--gamma", "1", "--lr", "0.1"],
            "l8_leakfree_lr15": L8 + LEAKFREE + ["--local-scale", "4", "--gamma", "1", "--cache-gamma", "0.98", "--lr", "0.15"],
            "l8_leakfree_cv_lr15": L8 + LEAKFREE_CV + ["--local-scale", "4", "--gamma", "1", "--cache-gamma", "0.98", "--lr", "0.15"],
        },
    },
    ".seed": {"values": [42], "flags": "--seed"},
}
METRIC = "val_second_half"
GOAL = "minimize"
