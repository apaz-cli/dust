# Does MoE's gap under Dust close with more draws at the MLP sites? Each expert weight sees only 1/4 of the tokens, so
# its estimate is noisier than a dense weight's; backprop has no such noise. Even without live units and the control
# variate (no expert bug), MoE trails dense by 0.32 under Dust with AdamW (moe_router_dust_adamw). MLP-input draws at
# 2x and 4x (writers unchanged), MoE with --moe-fix and dense, AdamW 3e-3, 32 sequences, 4 layers, 1500 steps, seed 42.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tinystories_long", "--steps", "1500",
           "--local-scale", "1", "--microbatch-size", "8", "--batch-seqs", "32", "--opt", "adamw", "--adam-lr", "3e-3",
           "--moe-fix"] + LEAKFREE_CV
H = {2: '{"hidden": [76, 64, 32, 20]}', 4: '{"hidden": [152, 128, 64, 40]}'}  # the recipe's 4-layer counts x2, x4
OPTIONS = {".arm": {"values": ["moe_h2", "moe_h4", "dense_h4"],
                    "flags": {"moe_h2": ["--moe", "8,2,384,256", "--alloc", H[2]],
                              "moe_h4": ["--moe", "8,2,384,256", "--alloc", H[4]],
                              "dense_h4": ["--alloc", H[4]]}}}
METRIC = "val_loss"
GOAL = "minimize"
