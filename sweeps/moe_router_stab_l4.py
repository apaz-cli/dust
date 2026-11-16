# Diagnosing the loss spikes in Dust MoE runs (gradient norm creeps up for ~25 steps, then the loss jumps on alternate
# steps). Every step now logs each router's logit size (router_rms) and the gradient norm per model part
# (grad_norm_part). Candidate stabilizers that need no backward formula: a soft cap on router logits (c*tanh(x/c),
# Dust's router site sits before the cap), and decoupled weight decay on the router only (keeps its logits from
# growing, the job a z-loss does). Dust with AdamW 3e-3, --moe-fix, clipping on, 8 experts, 32 sequences, 4 layers,
# 1500 steps, seed 42.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tinystories_long", "--steps", "1500",
           "--local-scale", "1", "--microbatch-size", "8", "--batch-seqs", "32", "--opt", "adamw", "--adam-lr", "3e-3",
           "--moe", "8,2,384,256", "--moe-fix"] + LEAKFREE_CV
OPTIONS = {".arm": {"values": ["base", "cap5", "cap2", "wd1", "wd10"],
                    "flags": {"base": [], "cap5": ["--router-cap", "5"], "cap2": ["--router-cap", "2"],
                              "wd1": ["--router-wd", "1"], "wd10": ["--router-wd", "10"]}}}
METRIC = "val_loss"
GOAL = "minimize"
