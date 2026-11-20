# NVFP4 reruns (--nvfp4: the reruns' linear layers as a delta forward on FP4 tensor cores, yoco/nvfp4.py), against
# the same runs without, on the current code (tuned expert kernel). fineweb-edu, 4 layers, SGD at the recipe's rate,
# 1500 steps. Quality: validation loss with and without; speed: seconds per step (on the B300, shared GPUs).
# - moe8_w512: 8 experts, width 512, 64 sequences, 4x writer draws (the scale_moe8_sgd setup).
# - dense_w1024: dense, width 1024, 32 sequences (on the RTX 5090 NVFP4 was 1.17x faster per step here).
# - moe8_w1024: 8 experts of 1536 plus a shared 1024 (active width 4096 = 4 x 1024), width 1024, 32 sequences.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE as _EX  # noqa: F401

UPLOAD_EXCLUDE = _EX + ["/ckpt", "/data/tinystories", "/data/tinystories_long", "/data/tsrepeat", "/data/tsrepeat2048",
                       "/data/tsrepeat8192", "/runs"]

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/fineweb_edu", "--n-layer", "4",
           "--steps", "1500", "--save-every", "0", "--local-scale", "4", "--microbatch-size", "16"] + LEAKFREE_CV
FIX = ["--moe-fix", "--router-cap", "2", "--router-wd", "10"]
SETUPS = {
    "moe8_w512": ["--n-embd", "512", "--batch-seqs", "64", "--moe", "8,2,768,512", "--alloc", '{"writers": [152, 128, 64, 40]}'] + FIX,
    "dense_w1024": ["--n-embd", "1024", "--batch-seqs", "32"],
    "moe8_w1024": ["--n-embd", "1024", "--batch-seqs", "32", "--moe", "8,2,1536,1024"] + FIX,
}
flags = {}
for name, f in SETUPS.items():
    flags[f"{name}_base"] = f
    flags[f"{name}_fp4"] = f + ["--nvfp4"]
OPTIONS = {".arm": {"values": list(flags), "flags": flags}}
METRIC = "val_loss"
GOAL = "minimize"
