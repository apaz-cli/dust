# Method 1 for the expert noise: the experts' hidden errors from local probes (--expert-probes N) instead of full
# reruns. Each probe jitters the live expert units, reruns only relu^2 and that expert's output projection (times its
# gate weight), and scores the change against the error at the MoE layer's output, which is estimated from all tokens.
# Each slot is scored on its own change. Forward only; antithetic pairs. 256 and 1024 probes. Compare moe_fix_l4
# fix_s32 (full draws, 2.710) and its expert oracle; dense 2.286. Dust with AdamW 3e-3, --moe-fix, clipping, 32
# sequences, 4 layers, 1500 steps, seed 42.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tinystories_long", "--steps", "1500",
           "--local-scale", "1", "--microbatch-size", "8", "--batch-seqs", "32", "--opt", "adamw", "--adam-lr", "3e-3",
           "--moe", "8,2,384,256", "--moe-fix"] + LEAKFREE_CV
OPTIONS = {".probes": {"values": [256, 1024], "flags": "--expert-probes"}}
METRIC = "val_loss"
GOAL = "minimize"
