# Multi-query associative recall (8 pairs per block, 64 keys, 64 values; only answers scored), backprop, 4 layers.
# Shows that the repo's SGD cannot learn it while AdamW can (Context use note). Chance is ln 64 = 4.16.
from _common import PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--method", "bp",
           "--data-dir", "/home/apaz/git/dust_experiments/data/mqar8_k64v64"]
OPTIONS = {
    ".opt": {
        "values": ["sgd_lr0.2", "sgd_lr1.5", "adamw_3e-3"],
        "flags": {"sgd_lr0.2": ["--lr", "0.2"], "sgd_lr1.5": ["--lr", "1.5"],
                  "adamw_3e-3": ["--opt", "adamw", "--adam-lr", "3e-3"]},
    },
}
METRIC = "val_loss"
GOAL = "minimize"
