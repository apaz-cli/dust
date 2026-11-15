# The MoE scaling questions under Dust with AdamW (MoE needs it: with backprop and SGD, MoE loses at every batch size,
# moe_batch_l4). Replaces the Dust-with-SGD arms of moe_experts_l4, moe_placement_l4 and moe_scale_next. AdamW 3e-3.
# experts: tokens per expert held at about 8k (8 experts at 64 sequences in moe_batch_adamw_l4, 16 at 128, 32 at 256).
# placement: MoE only in the cross-decoder on a 2-layer self-decoder vs everywhere, 64 sequences.
# best8: the best 8-layer Dust setup (2-layer self-decoder, readout, twice the draws), dense vs MoE in the cross-decoder.
# long: 4 layers, 64 sequences, 6000 steps. TinyStories (72k rows), seed 42; backprop references in the original sweeps.
from _common import L8, LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tinystories_long",
           "--microbatch-size", "8", "--opt", "adamw", "--adam-lr", "3e-3"] + LEAKFREE_CV
MOE = ["--moe", "8,2,384,256"]
L4 = ["--local-scale", "1", "--steps", "1500"]
BEST8 = L8 + ["--clip", "2", "--local-scale", "4", "--n-self", "2", "--readout", "--readout-warmup", "100",
              "--direct-scale", "2", "--emb-scale", "8", "--steps", "1500", "--batch-seqs", "64"]
P = L4 + ["--batch-seqs", "64", "--n-self", "2"]
OPTIONS = {".arm": {"values": ["experts_e16_s128", "experts_e32_s256", "experts_dense_s128", "experts_dense_s256",
                               "place_l4_upper", "place_l6_dense", "place_l6_all", "place_l6_upper",
                               "best8_dense", "best8_moe_upper", "long_dense", "long_moe"],
                    "flags": {"experts_e16_s128": L4 + ["--moe", "16,2,384,256", "--batch-seqs", "128"],
                              "experts_e32_s256": L4 + ["--moe", "32,2,384,256", "--batch-seqs", "256"],
                              "experts_dense_s128": L4 + ["--batch-seqs", "128"],
                              "experts_dense_s256": L4 + ["--batch-seqs", "256"],
                              "place_l4_upper": P + MOE + ["--moe-layers", "2", "3"],
                              "place_l6_dense": P + ["--n-layer", "6"],
                              "place_l6_all": P + ["--n-layer", "6"] + MOE,
                              "place_l6_upper": P + ["--n-layer", "6"] + MOE + ["--moe-layers", "2", "3", "4", "5"],
                              "best8_dense": BEST8,
                              "best8_moe_upper": BEST8 + MOE + ["--moe-layers", "2", "3", "4", "5", "6", "7"],
                              "long_dense": ["--local-scale", "1", "--steps", "6000", "--batch-seqs", "64"],
                              "long_moe": ["--local-scale", "1", "--steps", "6000", "--batch-seqs", "64"] + MOE}}}
METRIC = "val_loss"
GOAL = "minimize"
