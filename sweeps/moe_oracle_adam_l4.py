# Under AdamW, is it the expert weights' draw noise that breaks Dust MoE? (The router is not: a frozen router keeps
# the load even and MoE still stalls, moe_router_adam_l4.) Oracle runs (one group noise-free, derivatives as a
# measuring stick) with AdamW 3e-3, current recipe, MoE, 4-layer TinyStories, 1500 steps, seed 42.
from _common import LEAKFREE_CV, TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

COMMAND = command(TINYSTORIES) + LEAKFREE_CV + ["--local-scale", "1", "--steps", "1500", "--opt", "adamw",
                                                "--adam-lr", "3e-3", "--moe", "8,2,384,256"]
OPTIONS = {".oracle": {"values": ["experts", "all_mlp"],
                       "flags": {"experts": ["--oracle-re", r"mlp\.w_(fc|proj)"], "all_mlp": ["--oracle-re", r"\.mlp\."]}}}
METRIC = "val_loss"
GOAL = "minimize"
