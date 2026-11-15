# Where should the experts go under Dust? MoE only in the upper half (the cross-decoder, where Dust's credit is nearly
# exact) on a 2-layer self-decoder, against MoE in every layer and dense, at 64 sequences per step (about 8k tokens per
# expert). 4 layers (2 self-decoder + 2 cross-decoder) and 6 layers (2 + 4). Real Dust and backprop (AdamW). 1500
# steps, TinyStories (72k rows), seed 42.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tinystories_long", "--steps", "1500",
           "--batch-seqs", "64", "--n-self", "2"]
OPTIONS = {
    ".method": {"values": ["dust", "bp_adamw"],
                "flags": {"dust": LEAKFREE_CV + ["--local-scale", "1", "--microbatch-size", "8"],
                          "bp_adamw": ["--method", "bp", "--opt", "adamw", "--adam-lr", "3e-3"]}},
    ".cfg": {"values": ["l4_upper", "l6_dense", "l6_all", "l6_upper"],
             "flags": {"l4_upper": ["--moe", "8,2,384,256", "--moe-layers", "2", "3"],
                       "l6_dense": ["--n-layer", "6"],
                       "l6_all": ["--n-layer", "6", "--moe", "8,2,384,256"],
                       "l6_upper": ["--n-layer", "6", "--moe", "8,2,384,256", "--moe-layers", "2", "3", "4", "5"]}},
}
METRIC = "val_loss"
GOAL = "minimize"
