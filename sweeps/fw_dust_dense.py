# fineweb-edu, step 2: retune dense Dust. The recipe (leak-free scoring, live units, control variate), width 512,
# 4 layers, 64 sequences in microbatches of 16 (fits the RTX 4070 at this width), 1500 steps; AdamW at 1e-3, 3e-3 and
# 1e-2, and SGD at the recipe's rate for reference. Local draws x4 as at 8 layers and width 512.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

FW = ["--data-dir", "data/fineweb_edu", "--n-layer", "4", "--n-embd", "512", "--steps", "1500"]
COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--local-scale", "4", "--batch-seqs", "64",
           "--microbatch-size", "16"] + FW + LEAKFREE_CV
OPTIONS = {".opt": {"values": ["adamw1e-3", "adamw3e-3", "adamw1e-2", "sgd"],
                    "flags": {"adamw1e-3": ["--opt", "adamw", "--adam-lr", "1e-3"],
                              "adamw3e-3": ["--opt", "adamw", "--adam-lr", "3e-3"],
                              "adamw1e-2": ["--opt", "adamw", "--adam-lr", "1e-2"], "sgd": []}}}
METRIC = "val_loss"
GOAL = "minimize"
