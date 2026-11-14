# Better readout targets: the attention-output targets are chained from each block's output-projection error by
# local probes (--chain 256 in the recipe); 1024 probes make them less noisy (and also sharpen the attention-internal
# estimates, so the recipe at 1024 without the readout is the control). 4 layers seeds 42 and 7; 8 layers seed 42
# (readout with the 100-step warm-up).
from _common import L8, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

C = [a if a != "256" else "1024" for a in LEAKFREE_CV]
L4 = ["--local-scale", "1"]
L8A = ["--clip", "2"] + L8 + ["--local-scale", "4"]
COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--data-dir", TINYSTORIES] + C
OPTIONS = {".arm": {"values": ["l4_ro_seed42", "l4_ro_seed7", "l4_plain_seed42", "l4_plain_seed7", "l8_ro_seed42",
                               "l8_plain_seed42"],
                    "flags": {"l4_ro_seed42": L4 + ["--readout", "--seed", "42"],
                              "l4_ro_seed7": L4 + ["--readout", "--seed", "7"],
                              "l4_plain_seed42": L4 + ["--seed", "42"],
                              "l4_plain_seed7": L4 + ["--seed", "7"],
                              "l8_ro_seed42": L8A + ["--readout", "--readout-warmup", "100", "--seed", "42"],
                              "l8_plain_seed42": L8A + ["--seed", "42"]}}}
METRIC = "val_loss"
GOAL = "minimize"
