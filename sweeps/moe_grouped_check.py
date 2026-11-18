# The grouped-matmul MoE forward (--moe-grouped: one torch._grouped_mm per projection over the token slots sorted by
# expert; fixed shapes, so the model forward compiles), against the current sparse path, uncompiled. fineweb-edu,
# the scale_moe8_sgd setup of fw_dust_moe_scale, 30 steps: training losses should track and seconds per step drop.
# grouped_compiled keeps the compile self-check's decision; *_force keeps the compiled forward even if it fails (the
# check's worst-token difference may come from routing ties flipping, which the logged mean and share show).
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

FW = ["--data-dir", "data/fineweb_edu", "--n-layer", "4", "--n-embd", "512", "--steps", "30", "--eval-every", "10"]
COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--local-scale", "4", "--microbatch-size", "16",
           "--batch-seqs", "64", "--moe", "8,2,768,512", "--moe-fix", "--router-cap", "2", "--router-wd", "10",
           "--alloc", '{"writers": [152, 128, 64, 40]}'] + FW + LEAKFREE_CV
OPTIONS = {".arm": {"values": ["sparse_eager", "grouped_eager", "grouped_compiled", "grouped_force", "dense_force"],
                    "flags": {"sparse_eager": ["--no-compile"], "grouped_eager": ["--moe-grouped", "--no-compile"],
                              "grouped_compiled": ["--moe-grouped"], "grouped_force": ["--moe-grouped", "--compile-force"],
                              "dense_force": ["--moe-dense", "--compile-force"]}}}
METRIC = "train_loss"
GOAL = "minimize"
