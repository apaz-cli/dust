# Noise simulator: does a noise floor (noise on tokens whose own error is near zero, as leakage and curvature give
# real Dust) break AdamW the way real Dust does? Heavy tails did not (sim_noise_tails). Floor F = 1 and 10 times the
# mean per-token error, with C lowered so the total noise is about that of C = 1 without a floor (C = 0.01 also, less);
# SGD and AdamW 3e-3,
# 1500 steps, TinyStories, 4 layers.
from _common import PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--method", "ed", "--emb-scale", "4", "--wsd", "0.2",
           "--local-scale", "1", "--sim-cv", "ckpt/sim/cv_l4.json", "--gamma", "0.98",
           "--data-dir", TINYSTORIES, "--steps", "1500"]
OPTIONS = {
    ".floor": {"values": ["f1", "f10", "f10_c01"],
               "flags": {"f1": ["--sim-floor", "1", "--sim-noise", "0.5"], "f10": ["--sim-floor", "10", "--sim-noise", "0.1"],
                         "f10_c01": ["--sim-floor", "10", "--sim-noise", "0.01"]}},
    ".opt": {"values": ["sgd", "adamw"], "flags": {"sgd": [], "adamw": ["--opt", "adamw", "--adam-lr", "3e-3"]}},
}
METRIC = "val_loss"
GOAL = "minimize"
