# MoE scaling, next round. (a) The best 8-layer Dust setup (2-layer self-decoder, readout, twice the draws) with MoE in
# the cross-decoder against dense, 64 sequences per step (about 8k tokens per expert). (b) Longer training: dense vs MoE
# under Dust, 4 layers, 64 sequences, 6000 steps. TinyStories (72k rows; extra passes wrap around), seed 42.
from _common import L8, LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tinystories_long",
           "--batch-seqs", "64", "--microbatch-size", "8"] + LEAKFREE_CV
MOE = ["--moe", "8,2,384,256"]
BEST8 = L8 + ["--clip", "2", "--local-scale", "4", "--n-self", "2", "--readout", "--readout-warmup", "100",
              "--direct-scale", "2", "--emb-scale", "8", "--steps", "1500"]
L4 = ["--local-scale", "1"]
OPTIONS = {".arm": {"values": ["l8best_dense", "l8best_moe_upper", "l4_dense_6k", "l4_moe_6k"],
                    "flags": {"l8best_dense": BEST8,
                              "l8best_moe_upper": BEST8 + MOE + ["--moe-layers", "2", "3", "4", "5", "6", "7"],
                              "l4_dense_6k": L4 + ["--steps", "6000"],
                              "l4_moe_6k": L4 + MOE + ["--steps", "6000"]}}}
METRIC = "val_loss"
GOAL = "minimize"
