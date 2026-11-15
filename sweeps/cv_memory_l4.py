# The control variate's fit is a running average over past steps (--cv-beta 0.9, about 10 steps). If the fit's own
# errors, which persist from step to step, are why richer predictions hurt (cv_neighbor, cv_own), a longer memory
# (more data per fit, smaller errors) should help, most with the clean extra input. 4 layers, TinyStories, seed 42;
# the recipe at beta 0.9 reached 2.344, with --cv-own 2.423.
from _common import LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", TINYSTORIES, "--local-scale", "1"] + LEAKFREE_CV
OPTIONS = {".arm": {"values": ["b97", "b99", "own_b97", "own_b99"],
                    "flags": {"b97": ["--cv-beta", "0.97"], "b99": ["--cv-beta", "0.99"],
                              "own_b97": ["--cv-own", "--cv-beta", "0.97"], "own_b99": ["--cv-own", "--cv-beta", "0.99"]}}}
METRIC = "val_loss"
GOAL = "minimize"
