# fineweb-edu, round 3: hidden draws. Dense Dust gained nothing from 2, 4 or 8 times the writer draws (4.383, 4.398,
# 4.381 against 4.409) but 0.05 from 4 times the MLP-input (hidden) draws on top (levers_dense_sgd, 4.347). MoE at 8
# experts lost to dense by 0.085 with writer draws only (scale_moe8_sgd). Here: where the dense hidden-draw gain levels
# off (2, 4, 8 times), whether MoE gains more from hidden draws than dense does (8 experts, every layer, 4 and 8 times),
# and the scaling test with hidden draws (dense at 128 and 256 sequences, 16 experts at 128). All SGD, 4x writer draws,
# same base as fw_dust_moe_scale.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

FW = ["--data-dir", "data/fineweb_edu", "--n-layer", "4", "--n-embd", "512", "--steps", "1500"]
COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--local-scale", "4", "--microbatch-size", "16"] + FW + LEAKFREE_CV
WR = [152, 128, 64, 40]
BASE = [38, 32, 16, 10]
A = lambda h: ["--alloc", '{"writers": %s, "hidden": %s}' % (WR, [b * h for b in BASE])]
FIX = ["--moe-fix", "--router-cap", "2", "--router-wd", "10"]
S = lambda n: ["--batch-seqs", str(n)]
M = lambda e: ["--moe", f"{e},2,768,512"] + FIX
OPTIONS = {".arm": {"values": ["dense_h2", "dense_h8", "moe8_h4", "moe8_h8", "dense128_h4", "dense256_h4", "moe16_h4"],
                    "flags": {"dense_h2": S(64) + A(2), "dense_h8": S(64) + A(8),
                              "moe8_h4": S(64) + M(8) + A(4), "moe8_h8": S(64) + M(8) + A(8),
                              "dense128_h4": S(128) + A(4), "dense256_h4": S(256) + A(4),
                              "moe16_h4": S(128) + M(16) + A(4)}}}
METRIC = "val_loss"
GOAL = "minimize"
