# Is the optimizer why MoE loses under Dust? MoE beats dense under backprop only with AdamW, and Dust has only been
# run with SGD. Noise-free Dust (Dust's average, computed exactly) and real Dust (current recipe) with AdamW, dense
# vs MoE (vs wide dense for noise-free), 4-layer TinyStories, 1500 steps, seed 42.
from _common import LEAKFREE_CV, TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

COMMAND = command(TINYSTORIES) + ["--steps", "1500", "--opt", "adamw"]
OPTIONS = {
    ".method": {
        ".noisefree": {"flags": ["--method", "ed", "--gamma", "0.98"],
                       ".adam_lr": {"values": [1e-3, 3e-3], "flags": "--adam-lr"},
                       ".mlp": {"values": ["dense", "moe", "wide"],
                                "flags": {"dense": [], "moe": ["--moe", "8,2,384,256"], "wide": ["--moe", "1,1,3072,256"]}}},
        ".dust": {"flags": LEAKFREE_CV + ["--local-scale", "1"],
                  ".adam_lr": {"values": [3e-4, 1e-3, 3e-3], "flags": "--adam-lr"},
                  ".mlp": {"values": ["dense", "moe"],
                           "flags": {"dense": [], "moe": ["--moe", "8,2,384,256"]}}},
    },
}
METRIC = "val_loss"
GOAL = "minimize"
