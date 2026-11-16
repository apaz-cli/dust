# Orthogonal draws (noise PR #5) for the expert noise that keeps MoE behind dense under Dust (moe_fix_l4 oracles: with
# the routed experts noise-free MoE matches dense). Each token's draws at a site are distinct rows of a random
# orthogonal basis, so no direction is measured twice; the estimate becomes exact when the draws reach the site's
# width. Expected: small gain at the recipe's draw counts (about 38 against 768 expert units per token), larger at 16x
# (about 600). MoE (--moe-fix) with orthogonal draws at the expert site, at the expert and shared-expert sites, and at the
# expert site with 16x MLP draws (compare moe_hidden_draws_l4 moe_h16); dense with orthogonal MLP-input draws.
# Dust with AdamW 3e-3, clipping on, 32 sequences, 4 layers, 1500 steps, seed 42.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tinystories_long", "--steps", "1500",
           "--local-scale", "1", "--microbatch-size", "8", "--batch-seqs", "32", "--opt", "adamw", "--adam-lr", "3e-3",
           "--moe-fix"] + LEAKFREE_CV
MOE = ["--moe", "8,2,384,256"]
H16 = '{"hidden": [608, 512, 256, 160]}'
OPTIONS = {".arm": {"values": ["moe_orth_experts", "moe_orth_mlp", "moe_h16_orth", "dense_orth"],
                    "flags": {"moe_orth_experts": MOE + ["--orth", "mlp.moe"],
                              "moe_orth_mlp": MOE + ["--orth", "mlp.moe", "mlp.c_fc"],
                              "moe_h16_orth": MOE + ["--orth", "mlp.moe", "--alloc", H16],
                              "dense_orth": ["--orth", "mlp.c_fc"]}}}
METRIC = "val_loss"
GOAL = "minimize"
