# Plan item 4, first step toward a learned control variate: each site's prediction also uses a nearby site's error
# estimate from the same step (separate draws, made first: the step goes top-down). Unbiased; the ridge fit and the
# trust scale decide how much it is used. The recipe with --cv-neighbor; baselines: headline_tinystories_l4,
# readout_l4 (copy), readout_l8. Logs cv_explained per site group (the share of the draws' variance removed).
from _common import COPY_TASK, L8, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--cv-neighbor"] + LEAKFREE_CV
OPTIONS = {".arm": {"values": ["l4_ts", "l4_copy", "l8_ts"],
                    "flags": {"l4_ts": ["--local-scale", "1", "--data-dir", TINYSTORIES],
                              "l4_copy": ["--local-scale", "1", "--data-dir", COPY_TASK, "--gamma", "1",
                                          "--cache-gamma", "0.98"],
                              "l8_ts": ["--clip", "2", "--local-scale", "4", "--data-dir", TINYSTORIES] + L8}}}
METRIC = "val_loss"
GOAL = "minimize"
