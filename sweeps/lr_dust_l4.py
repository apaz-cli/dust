# Learning-rate sweep for Dust (current recipe: leak-free scoring + live units + control variate, 1x local
# draws), dense vs MoE, 4-layer TinyStories, 1500 steps, seed 42. Dust's lr has never been tuned for our recipes
# (0.25 came from the Dust repo). moe_m4: expert weights get 4x the lr (each expert sees 2/8 of the tokens).
from _common import LEAKFREE_CV, TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

COMMAND = command(TINYSTORIES) + LEAKFREE_CV + ["--local-scale", "1", "--steps", "1500"]
OPTIONS = {
    ".mlp": {"values": ["dense", "moe", "moe_m4"],
             "flags": {"dense": [], "moe": ["--moe", "8,2,384,256"],
                       "moe_m4": ["--moe", "8,2,384,256", "--moe-lr-mult", "4"]}},
    ".lr": {"values": [0.1, 0.15, 0.25, 0.35, 0.5], "flags": "--lr"},
    ".seed": {"values": [42], "flags": "--seed"},
}
METRIC = "val_loss"
GOAL = "minimize"
