# The grouped MoE forward on the B300 (sm100): the first version used torch._grouped_mm, whose sm100 kernel fails to
# launch in the torch 2.10 wheels; now a Triton kernel (yoco/gmm.py). 30 steps of the 8-expert fineweb-edu setup,
# compiled (the default) and eager; the training losses should match sparse_eager of moe_grouped_check_20261117_2356_1b8c
# (6.4133 at step 30) and the compile self-check pass with the routing pinned.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

FW = ["--data-dir", "data/fineweb_edu", "--n-layer", "4", "--n-embd", "512", "--steps", "30", "--eval-every", "10"]
COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--local-scale", "4", "--microbatch-size", "16",
           "--batch-seqs", "64", "--moe", "8,2,768,512", "--moe-fix", "--router-cap", "2", "--router-wd", "10",
           "--alloc", '{"writers": [152, 128, 64, 40]}'] + FW + LEAKFREE_CV
OPTIONS = {".arm": {"values": ["grouped_compiled", "grouped_eager"],
                    "flags": {"grouped_compiled": [], "grouped_eager": ["--no-compile"]}}}
METRIC = "train_loss"
GOAL = "minimize"
