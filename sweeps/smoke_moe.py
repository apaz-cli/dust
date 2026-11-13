# Short check that Dust on MoE runs with the current recipe (leak-free scoring + live units + control variate).
from _common import LEAKFREE_CV, TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

COMMAND = command(TINYSTORIES) + LEAKFREE_CV + ["--local-scale", "1", "--steps", "40", "--eval-every", "20"]
OPTIONS = {".mlp": {"values": ["moe", "moe_m4"],
                    "flags": {"moe": ["--moe", "8,2,384,256"], "moe_m4": ["--moe", "8,2,384,256", "--moe-lr-mult", "4"]}}}
METRIC = "val_loss"
GOAL = "minimize"
