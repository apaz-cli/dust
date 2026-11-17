# fineweb-edu, round 2: the rest of the 16 November plan (items 4 to 6 of STATUS "Next", plus all levers at 128
# sequences). Width 512, 1500 steps, recipe with every MoE fix (--moe-fix, router cap 2, router weight decay 10),
# 8 experts (2 active of 768 plus a shared 512). Writer counts per layer are the recipe's [38, 32, 16, 10, 6, 4, 2, 2].
# writers_*: where the writer-draw gain levels off, 4 layers, 64 sequences: writers 1x, 2x, 8x for MoE (SGD, AdamW)
#   and dense (SGD; dense 1x is fw_dust_dense, 4x is fw_dust_moe_scale).
# oracle_l*: which MoE layer's noise costs most, 4 layers, MoE everywhere, 4x writers, AdamW: one layer's experts
#   get their noise-free expected update (a measuring stick); no-oracle baseline is fw_dust_moe_scale scale_moe8_adam.
# place8_*: lower against upper half at 8 layers (4-layer self-decoder), 4x writers, AdamW; dense for reference.
# best8_*: MoE on the best 8-layer setup (2-layer self-decoder, readout, twice the draws), MoE everywhere or in the
#   self-decoder only, SGD and AdamW; dense with each optimizer.
# levers128_*: all levers (lower half, 4x writers and MLP inputs, per-expert control variate) at 128 sequences.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/fineweb_edu", "--n-embd", "512",
           "--steps", "1500", "--local-scale", "4"] + LEAKFREE_CV
BASE = [38, 32, 16, 10, 6, 4, 2, 2]
w = lambda L, k, h=None: ["--alloc", '{"writers": %s%s}' % ([v * k for v in BASE[:L]],
                                                           '' if h is None else ', "hidden": %s' % [v * h for v in BASE[:L]])]
L4 = ["--n-layer", "4", "--microbatch-size", "16"]
L8 = ["--n-layer", "8", "--microbatch-size", "8"]
S = lambda n: ["--batch-seqs", str(n)]
MOE = ["--moe", "8,2,768,512", "--moe-fix", "--router-cap", "2", "--router-wd", "10"]
ADAM = ["--opt", "adamw", "--adam-lr", "3e-3"]
BEST8 = ["--n-self", "2", "--readout", "--readout-warmup", "800", "--direct-scale", "2", "--emb-scale", "8"]
F = {}
for k in (1, 2, 8):
    F[f"writers{k}_moe_sgd"] = L4 + S(64) + MOE + w(4, k)
    F[f"writers{k}_moe_adam"] = L4 + S(64) + MOE + w(4, k) + ADAM
    if k != 1:
        F[f"writers{k}_dense_sgd"] = L4 + S(64) + w(4, k)
for l in range(4):
    F[f"oracle_l{l}"] = L4 + S(64) + MOE + w(4, 4) + ADAM + ["--oracle-re", rf"h\.{l}\.mlp\.w_(fc|proj)"]
F["place8_lower"] = L8 + S(64) + MOE + w(8, 4) + ADAM + ["--moe-layers", "0", "1", "2", "3"]
F["place8_upper"] = L8 + S(64) + MOE + w(8, 4) + ADAM + ["--moe-layers", "4", "5", "6", "7"]
F["place8_all"] = L8 + S(64) + MOE + w(8, 4) + ADAM
F["place8_dense"] = L8 + S(64) + w(8, 4) + ADAM
for o, extra in (("sgd", []), ("adam", ADAM)):
    F[f"best8_moe_all_{o}"] = L8 + S(64) + BEST8 + MOE + extra
    F[f"best8_moe_lower_{o}"] = L8 + S(64) + BEST8 + MOE + extra + ["--moe-layers", "0", "1"]
    F[f"best8_dense_{o}"] = L8 + S(64) + BEST8 + extra
LEV = MOE + w(4, 4, 4) + ["--moe-layers", "0", "1", "--moe-cv"]
F["levers128_moe_sgd"] = L4 + S(128) + LEV
F["levers128_moe_adam"] = L4 + S(128) + LEV + ADAM
F["levers128_dense_sgd"] = L4 + S(128) + w(4, 4, 4)
OPTIONS = {".arm": {"values": list(F), "flags": F}}
METRIC = "val_loss"
GOAL = "minimize"
