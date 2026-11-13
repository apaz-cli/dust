# Does backprop MoE beat dense with longer training? AdamW 3e-3 (best in lr_bp_l4), 3000 steps, 2 seeds.
from _common import TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

COMMAND = command(TINYSTORIES) + ["--method", "bp", "--opt", "adamw", "--adam-lr", "3e-3"]
OPTIONS = {
    ".mlp": {"values": ["dense", "moe", "wide"],
             "flags": {"dense": [], "moe": ["--moe", "8,2,384,256"], "wide": ["--moe", "1,1,3072,256"]}},
    ".seed": {"values": [42, 7], "flags": "--seed"},
}
METRIC = "val_loss"
GOAL = "minimize"
