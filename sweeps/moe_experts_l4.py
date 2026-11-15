# Does Dust's gain from MoE grow with the number of experts, as backprop's does? Tokens per expert held at about 8k per
# step (2 active experts per token): 8 experts at 64 sequences, 16 at 128, 32 at 256. Experts of 384 hidden units plus a
# shared expert of 256, so active size is fixed and total size grows. Dense at the same batch sizes as the reference.
# Real Dust (the recipe, SGD, microbatches of 8) and backprop (SGD, AdamW). 4 layers, 1500 steps, TinyStories (72k
# rows; 256 x 1500 sequences is about 5 passes), seed 42.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tinystories_long", "--steps", "1500"]
OPTIONS = {
    ".method": {"values": ["dust", "bp_sgd", "bp_adamw"],
                "flags": {"dust": LEAKFREE_CV + ["--local-scale", "1", "--microbatch-size", "8"],
                          "bp_sgd": ["--method", "bp", "--lr", "0.2"],
                          "bp_adamw": ["--method", "bp", "--opt", "adamw", "--adam-lr", "3e-3"]}},
    ".cfg": {"values": ["e16_s128", "e32_s256", "dense_s128", "dense_s256"],
             "flags": {"e16_s128": ["--moe", "16,2,384,256", "--batch-seqs", "128"],
                       "e32_s256": ["--moe", "32,2,384,256", "--batch-seqs", "256"],
                       "dense_s128": ["--batch-seqs", "128"],
                       "dense_s256": ["--batch-seqs", "256"]}},
}
METRIC = "val_loss"
GOAL = "minimize"
