# Is the compiled head-draw chunk correct on the B300s? (Inductor gave wrong rerun estimates there, nan_debug_b300.)
# MoE config (only the head draws and scoring are compiled for MoE), 2 steps, compiled vs not; compare grad_norm_part.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tinystories_long", "--local-scale", "1",
           "--microbatch-size", "32", "--batch-seqs", "32", "--opt", "adamw", "--adam-lr", "3e-3", "--steps", "2",
           "--eval-every", "1", "--moe", "8,2,384,256", "--moe-fix"] + LEAKFREE_CV
OPTIONS = {".arm": {"values": ["compiled", "eager"], "flags": {"compiled": [], "eager": ["--no-compile"]}}}
METRIC = "val_loss"
GOAL = "minimize"
