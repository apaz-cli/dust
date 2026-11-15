# Plan item 4: a learned control variate with clean inputs. Each site's prediction also uses the clean stream entering
# its block (from the clean pass, so no noise), besides the head-error key. The neighbor version (noisy inputs) was
# worse (cv_neighbor). Baselines: headline_tinystories_l4, readout_l4 (copy), readout_l8. Logs cv_explained.
from _common import COPY_TASK, L8, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--cv-own"] + LEAKFREE_CV
OPTIONS = {".arm": {"values": ["l4_ts", "l4_copy", "l8_ts"],
                    "flags": {"l4_ts": ["--local-scale", "1", "--data-dir", TINYSTORIES],
                              "l4_copy": ["--local-scale", "1", "--data-dir", COPY_TASK, "--gamma", "1",
                                          "--cache-gamma", "0.98"],
                              "l8_ts": ["--clip", "2", "--local-scale", "4", "--data-dir", TINYSTORIES] + L8}}}
METRIC = "val_loss"
GOAL = "minimize"
