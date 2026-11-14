# Short check that the cross-token readout trains (and its step time) with the current recipe.
from _common import LEAKFREE_CV, TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

COMMAND = command(TINYSTORIES) + LEAKFREE_CV + ["--local-scale", "1", "--steps", "60", "--eval-every", "30"]
OPTIONS = {".readout": {"values": [0, 1], "flags": {0: [], 1: ["--readout"]}}}
METRIC = "val_loss"
GOAL = "minimize"
