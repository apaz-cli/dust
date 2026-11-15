# Plan item 1, the kill test: how much does backprop lose with a shallow self-decoder (YOCO's lower half) at the same
# total depth? 4 layers with 1, 2 (default), 3 self-decoder layers; 8 layers with 1, 2, 4 (default), 6. TinyStories
# and the copy task with SGD (lr 0.2, as the headline backprop runs) and AdamW (3e-3); associative recall with AdamW
# only (SGD stays at chance there). Seeds 42 and 7.
from _common import COPY_TASK, L8, MQAR, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--method", "bp"]
OPTIONS = {
    ".L": {"values": [4, 8], "flags": {4: [], 8: L8}},
    ".nself": {"values": [1, 2, 3, 4, 6], "flags": "--n-self"},
    ".task": {"values": ["ts_sgd", "ts_adamw", "copy_sgd", "copy_adamw", "mqar_adamw"],
              "flags": {"ts_sgd": ["--data-dir", TINYSTORIES, "--lr", "0.2"],
                        "ts_adamw": ["--data-dir", TINYSTORIES, "--opt", "adamw", "--adam-lr", "3e-3"],
                        "copy_sgd": ["--data-dir", COPY_TASK, "--lr", "0.2"],
                        "copy_adamw": ["--data-dir", COPY_TASK, "--opt", "adamw", "--adam-lr", "3e-3"],
                        "mqar_adamw": ["--data-dir", MQAR, "--opt", "adamw", "--adam-lr", "3e-3"]}},
    ".seed": {"values": [42, 7], "flags": "--seed"},
}


def EXCLUDE(combo):
    return combo["nself"] not in ({1, 2, 3} if combo["L"] == 4 else {1, 2, 4, 6})


METRIC = "val_loss"
GOAL = "minimize"
