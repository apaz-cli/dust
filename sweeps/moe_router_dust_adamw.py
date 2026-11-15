# Under Dust with AdamW the experts' load goes very uneven early (busiest expert 3.5x its share at 8 experts, 6-7x at
# 16; backprop with AdamW stays at 1.03-1.13; moe_batch_adamw_l4, moe_dust_adamw). Diagnostics at 32 sequences, AdamW
# 3e-3, 8 experts: load balancing 10x faster; router learning rate 0.1x; and without live units and the control
# variate (the one measured AdamW interaction, adam_dust_cause), dense and MoE. 4 layers, 1500 steps, seed 42.
from _common import LEAKFREE, LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tinystories_long", "--steps", "1500",
           "--local-scale", "1", "--microbatch-size", "8", "--batch-seqs", "32", "--opt", "adamw", "--adam-lr", "3e-3"]
MOE = ["--moe", "8,2,384,256"]
OPTIONS = {".arm": {"values": ["balance10", "router01", "nocv_dense", "nocv_moe"],
                    "flags": {"balance10": LEAKFREE_CV + MOE + ["--moe-balance-rate", "1e-2"],
                              "router01": LEAKFREE_CV + MOE + ["--router-lr-mult", "0.1"],
                              "nocv_dense": LEAKFREE,
                              "nocv_moe": LEAKFREE + MOE}}}
METRIC = "val_loss"
GOAL = "minimize"
