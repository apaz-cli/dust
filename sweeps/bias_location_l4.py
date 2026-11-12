# Where Dust's bias lives: noise-free Dust (derivatives as a measuring stick) with full credit from later tokens
# everywhere, except that the chosen layers keep plain Dust's own-token target. 4-layer TinyStories, seed 42.
# The cost of each choice is its loss minus the "none" arm's.
from _common import TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

COMMAND = command(TINYSTORIES) + ["--method", "ed", "--gamma", "1", "--lr", "0.2"]
OPTIONS = {
    ".own_layers": {
        "values": ["none", "0", "1", "0_1", "2_3", "all"],
        "flags": {"none": ["--ed-own-layers"], "0": ["--ed-own-layers", "0"], "1": ["--ed-own-layers", "1"],
                  "0_1": ["--ed-own-layers", "0", "1"], "2_3": ["--ed-own-layers", "2", "3"],
                  "all": ["--ed-own-layers", "0", "1", "2", "3"]},
    },
}
METRIC = "val_loss"
GOAL = "minimize"
