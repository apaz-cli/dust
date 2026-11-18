# The grouped-matmul MoE forward (--moe-grouped: one torch._grouped_mm per projection over the token slots sorted by
# expert; fixed shapes, so the model forward compiles), against the current sparse path, uncompiled. fineweb-edu,
# the scale_moe8_sgd setup of fw_dust_moe_scale, 30 steps: training losses should track and seconds per step drop.
# The first try (moe_grouped_check_20261117_2356_1b8c) never ran the grouped path: fast16 runs in fp16 and the path
# required bf16. Now the two expert matmuls run in bf16 (torch._grouped_mm takes nothing else) and the self-check
# compares compiled and eager with the routing pinned, logging separately how many tokens flip experts when it is free.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

FW = ["--data-dir", "data/fineweb_edu", "--n-layer", "4", "--n-embd", "512", "--steps", "30", "--eval-every", "10"]
COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--local-scale", "4", "--microbatch-size", "16",
           "--batch-seqs", "64", "--moe", "8,2,768,512", "--moe-fix", "--router-cap", "2", "--router-wd", "10",
           "--alloc", '{"writers": [152, 128, 64, 40]}'] + FW + LEAKFREE_CV
OPTIONS = {".arm": {"values": ["grouped_compiled", "grouped_eager", "dense_compiled"],
                    "flags": {"grouped_compiled": ["--moe-grouped"], "grouped_eager": ["--moe-grouped", "--no-compile"],
                              "dense_compiled": ["--moe-dense"]}}}
METRIC = "train_loss"
GOAL = "minimize"
