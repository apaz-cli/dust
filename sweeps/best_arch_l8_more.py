# Follow-ups to best_arch_l8 (2-layer self-decoder plus the readout, the best 8-layer copy result so far): a 1-layer
# self-decoder with the readout, and the 2-layer version with twice the residual-stream draws as well (at 4 layers the
# readout and twice the draws add up). TinyStories and the copy task, seed 42.
from _common import COPY_TASK, L8, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--clip", "2", "--local-scale", "4",
           "--readout", "--readout-warmup", "100"] + L8 + LEAKFREE_CV
TS = ["--data-dir", TINYSTORIES]
COPY = ["--data-dir", COPY_TASK, "--gamma", "1", "--cache-gamma", "0.98", "--lr", "0.15"]
X2 = ["--direct-scale", "2", "--emb-scale", "8"]
OPTIONS = {".arm": {"values": ["n1_ts", "n1_copy", "n2x2_ts", "n2x2_copy"],
                    "flags": {"n1_ts": ["--n-self", "1"] + TS, "n1_copy": ["--n-self", "1"] + COPY,
                              "n2x2_ts": ["--n-self", "2"] + X2 + TS, "n2x2_copy": ["--n-self", "2"] + X2 + COPY}}}
METRIC = "val_loss"
GOAL = "minimize"
