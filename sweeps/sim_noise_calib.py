# Plan item 7: the noise simulator. Noise-free Dust (method ed) with Gaussian noise added to each site's error, of
# size sqrt(C d / N) per token (d the site's width, N its draws in the recipe). Calibrate C against real Dust: the
# recipe reaches 2.344 at 4 layers (headline_tinystories_l4) and 2.344 at 8 layers (readout_l8). Same draw counts
# and learning rate as the recipe. TinyStories, seed 42.
from _common import L8, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--method", "ed", "--gamma", "0.98",
           "--data-dir", TINYSTORIES, "--emb-scale", "4", "--wsd", "0.2"]
OPTIONS = {
    ".L": {"values": [4, 8], "flags": {4: ["--local-scale", "1"], 8: L8 + ["--local-scale", "4"]}},
    ".C": {"values": [0.25, 1, 4, 16], "flags": "--sim-noise"},
}
METRIC = "val_loss"
GOAL = "minimize"
