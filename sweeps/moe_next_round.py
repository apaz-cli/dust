# Next round on MoE under Dust (the gap to dense is the routed experts' draw noise, moe_fix_l4). All with --moe-fix, the
# router stabilizers (logit cap 2, router weight decay 10: no spikes, moe_router_stab_l4), clipping, AdamW 3e-3, 4 layers
# (2-layer self-decoder + 2-layer cross-decoder), 8 experts (2 active) + shared expert, TinyStories (72k rows), seed 42.
# (a) A control variate fitted per expert (--moe-cv), 32 and 64 sequences (32 sequences without it: 2.580).
# (b) More tokens per expert: 128 sequences (1500 steps), and 64 sequences for 6000 steps; dense at the same.
# (c) Where the experts go, 64 sequences: lower half only (self-decoder), upper half only (cross-decoder), everywhere; dense.
# (d) More writer draws, 32 sequences: the experts' output weights are estimated from the MoE layer's output error,
# which comes from the writer draws; more MLP draws alone stopped helping past 4x (moe_hidden_draws_l4: 2.537 at 4x,
# 2.547 at 8x). Writers 4x, and writers 4x with MLP inputs 4x.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tinystories_long",
           "--local-scale", "1", "--microbatch-size", "8", "--opt", "adamw", "--adam-lr", "3e-3", "--moe-fix"] + LEAKFREE_CV
MOE = ["--moe", "8,2,384,256", "--router-cap", "2", "--router-wd", "10"]
S32, S64, S128 = (["--batch-seqs", str(n), "--steps", "1500"] for n in (32, 64, 128))
L64 = ["--batch-seqs", "64", "--steps", "6000"]
OPTIONS = {".arm": {"values": ["cv_s32", "cv_s64", "moe_s128", "dense_s128", "moe_s64_6k", "dense_s64_6k",
                               "lower_s64", "upper_s64", "all_s64", "dense_s64", "w4_s32", "w4h4_s32"],
                    "flags": {"cv_s32": MOE + S32 + ["--moe-cv"], "cv_s64": MOE + S64 + ["--moe-cv"],
                              "moe_s128": MOE + S128, "dense_s128": S128,
                              "moe_s64_6k": MOE + L64, "dense_s64_6k": L64,
                              "lower_s64": MOE + S64 + ["--moe-layers", "0", "1"],
                              "upper_s64": MOE + S64 + ["--moe-layers", "2", "3"],
                              "all_s64": MOE + S64, "dense_s64": S64,
                              "w4_s32": MOE + S32 + ["--alloc", '{"writers": [152, 128, 64, 40]}'],
                              "w4h4_s32": MOE + S32 + ["--alloc", '{"writers": [152, 128, 64, 40], "hidden": [152, 128, 64, 40]}']}}}
METRIC = "val_loss"
GOAL = "minimize"
