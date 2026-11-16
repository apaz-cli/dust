# Dust MoE bug: live units were not applied to the routed experts (only to the shared expert), so inactive expert units
# got pure noise; and the control variate ran at the expert and router sites, whose units belong to a different expert
# for each token. --moe-fix corrects both. Dust with AdamW 3e-3, 8 experts, 32 and 64 sequences (dense references in
# moe_batch_adamw_l4: 2.286 and 2.148). With the fix at 32 sequences, oracle runs (that group's noise-free update,
# derivatives as a measuring stick): routed experts, shared expert, router. 4 layers, 1500 steps, seed 42.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tinystories_long", "--steps", "1500",
           "--local-scale", "1", "--microbatch-size", "8", "--opt", "adamw", "--adam-lr", "3e-3", "--moe", "8,2,384,256",
           "--moe-fix"] + LEAKFREE_CV
OPTIONS = {".arm": {"values": ["fix_s32", "fix_s64", "oracle_experts", "oracle_shared", "oracle_router"],
                    "flags": {"fix_s32": ["--batch-seqs", "32"], "fix_s64": ["--batch-seqs", "64"],
                              "oracle_experts": ["--batch-seqs", "32", "--oracle-re", r"mlp\.w_(fc|proj)"],
                              "oracle_shared": ["--batch-seqs", "32", "--oracle-re", r"mlp\.c_(fc|proj)"],
                              "oracle_router": ["--batch-seqs", "32", "--oracle-re", r"mlp\.router"]}}}
METRIC = "val_loss"
GOAL = "minimize"
