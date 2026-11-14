# Variants of the cross-token readout on 4-layer TinyStories, seed 42 (compare readout_l4: recipe 2.344, +readout 2.269):
# its weight in the reward (--readout-scale 0.5 / 2), and without the control variate (does it need the CV's better
# targets? compare headline_tinystories_l4's leakfree arm).
from _common import LEAKFREE, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--data-dir", TINYSTORIES, "--local-scale", "1", "--seed", "42"]
OPTIONS = {".arm": {"values": ["scale0.5", "scale2", "no_cv", "no_cv_plain"],
                    "flags": {"scale0.5": LEAKFREE_CV + ["--readout", "--readout-scale", "0.5"],
                              "scale2": LEAKFREE_CV + ["--readout", "--readout-scale", "2"],
                              "no_cv": LEAKFREE + ["--readout"],
                              "no_cv_plain": LEAKFREE}}}
METRIC = "val_loss"
GOAL = "minimize"
