# MoE needs AdamW: with backprop, MoE beats dense only with AdamW and loses with SGD at every batch size (moe_batch_l4).
# So Dust with AdamW, dense against MoE, at 32 and 64 sequences per step (4k and 8k tokens per expert), learning rates
# 1e-3 and 3e-3. 4 layers, 1500 steps, TinyStories (72k rows), seed 42.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tinystories_long", "--steps", "1500",
           "--local-scale", "1", "--microbatch-size", "8", "--opt", "adamw"] + LEAKFREE_CV
OPTIONS = {
    ".mlp": {"values": ["dense", "moe"], "flags": {"dense": [], "moe": ["--moe", "8,2,384,256"]}},
    ".seqs": {"values": [32, 64], "flags": "--batch-seqs"},
    ".lr": {"values": [1e-3, 3e-3], "flags": "--adam-lr"},
}
METRIC = "val_loss"
GOAL = "minimize"
