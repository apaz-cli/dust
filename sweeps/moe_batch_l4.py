# Does MoE beat dense under Dust once each expert sees a realistic number of tokens per step? Earlier MoE tests gave each
# expert about 1k tokens per step (8 sequences of 512, 8 experts, 2 active). Here 8, 32 and 64 sequences per step (up to
# 8k tokens per expert), on the 72k-row TinyStories set (same validation rows as data/tinystories), 1500 steps.
# Real Dust (the recipe, SGD, microbatches of 8 sequences) and backprop (SGD and AdamW) as references. 4 layers, seed 42.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tinystories_long", "--steps", "1500"]
OPTIONS = {
    ".method": {"values": ["dust", "bp_sgd", "bp_adamw"],
                "flags": {"dust": LEAKFREE_CV + ["--local-scale", "1", "--microbatch-size", "8"],
                          "bp_sgd": ["--method", "bp", "--lr", "0.2"],
                          "bp_adamw": ["--method", "bp", "--opt", "adamw", "--adam-lr", "3e-3"]}},
    ".mlp": {"values": ["dense", "moe"], "flags": {"dense": [], "moe": ["--moe", "8,2,384,256"]}},
    ".seqs": {"values": [8, 32, 64], "flags": "--batch-seqs"},
}
METRIC = "val_loss"
GOAL = "minimize"
