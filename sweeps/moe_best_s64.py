# The best MoE setup so far against dense at 64 sequences per step: --moe-fix, router logit cap 2 and router weight
# decay 10 (no spikes, 2.580 at 32 sequences, moe_router_stab_l4), with the recipe's and 4x the MLP draws; dense with 4x
# MLP draws (dense with the recipe's draws at 64 sequences: 2.148, moe_batch_adamw_l4, unclipped). Dust with AdamW 3e-3,
# clipping on, 4 layers, 1500 steps, seed 42.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tinystories_long", "--steps", "1500",
           "--local-scale", "1", "--microbatch-size", "8", "--batch-seqs", "64", "--opt", "adamw", "--adam-lr", "3e-3",
           "--moe-fix"] + LEAKFREE_CV
MOE = ["--moe", "8,2,384,256", "--router-cap", "2", "--router-wd", "10"]
H4 = ["--alloc", '{"hidden": [152, 128, 64, 40]}']
OPTIONS = {".arm": {"values": ["moe", "moe_h4", "dense_h4", "dense"],
                    "flags": {"moe": MOE, "moe_h4": MOE + H4, "dense_h4": H4, "dense": []}}}
METRIC = "val_loss"
GOAL = "minimize"
