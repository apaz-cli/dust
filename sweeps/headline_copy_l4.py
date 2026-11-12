# Copy task (each row: 256 story tokens, then the same 256 again), 4-layer YOCO, 3000 steps, 2 seeds.
# The copied half (val_second_half) can only be predicted by looking 256 tokens back.
# Long credit horizon (gamma 1) in the self-decoder, 0.98 on the cache, for the leak-free arms.
from _common import BEST, COPY_TASK, LEAKFREE, LEAKFREE_CV, UPLOAD_EXCLUDE, command  # noqa: F401

LONG = ["--gamma", "1", "--cache-gamma", "0.98", "--local-scale", "1"]
COMMAND = command(COPY_TASK)
OPTIONS = {
    ".arm": {
        "values": ["backprop", "noisefree", "best", "leakfree", "leakfree_cv"],
        "flags": {
            "backprop": ["--method", "bp", "--lr", "0.2"],
            "noisefree": ["--method", "ed", "--gamma", "1", "--lr", "0.2"],
            "best": BEST + ["--local-scale", "4"],
            "leakfree": LEAKFREE + LONG,
            "leakfree_cv": LEAKFREE_CV + LONG,
        },
    },
    ".seed": {"values": [42, 7], "flags": "--seed"},
}
METRIC = "val_second_half"
GOAL = "minimize"
