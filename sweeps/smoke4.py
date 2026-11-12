# Four short runs, so at least two land on the remote worker (only 2 local GPUs).
from _common import BEST, TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

COMMAND = command(TINYSTORIES) + BEST + ["--local-scale", "1", "--steps", "60", "--eval-every", "30"]
OPTIONS = {".seed": {"values": [1, 2, 3, 4], "flags": "--seed"}}
METRIC = "val_loss"
GOAL = "minimize"
