# One short run to check logging, the run folder and the upload size before the headline sweeps.
from _common import BEST, TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

COMMAND = command(TINYSTORIES) + BEST + ["--local-scale", "1", "--steps", "60", "--eval-every", "30"]
OPTIONS = {".seed": {"values": [42], "flags": "--seed"}}
METRIC = "val_loss"
GOAL = "minimize"
