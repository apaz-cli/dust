# Plan item 7: the noise simulator. Noise-free Dust (method ed) with Gaussian noise added to each site's weight
# error (not passed on to lower layers, as Dust's sites are estimated separately), of size sqrt(C d (1 - R2) / N) per
# token: d the site's width, N its draws in the recipe, R2 the share the control variate removed in the recipe run
# (ckpt/sim/cv_l4.json from headline_tinystories_l4 leakfree_cv seed 42, cv_l8.json from readout_l8 without the
# readout). Calibrate C against real Dust: the recipe reaches 2.344 at 4 and at 8 layers. TinyStories, seed 42.
from _common import L8, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--method", "ed", "--gamma", "0.98",
           "--data-dir", TINYSTORIES, "--emb-scale", "4", "--wsd", "0.2"]
OPTIONS = {
    ".L": {"values": [4, 8], "flags": {4: ["--local-scale", "1", "--sim-cv", "ckpt/sim/cv_l4.json"],
                                       8: L8 + ["--local-scale", "4", "--clip", "2", "--sim-cv", "ckpt/sim/cv_l8.json"]}},
    ".C": {"values": [0.03, 0.1, 0.3, 1], "flags": "--sim-noise"},
}
METRIC = "val_loss"
GOAL = "minimize"
