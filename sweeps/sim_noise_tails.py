# Noise simulator: does heavy-tailed noise (same variance) break AdamW the way real Dust does? With Gaussian noise
# AdamW beat SGD (sim_noise_more); real Dust is the reverse. Student-t noise with 3 or 2.5 degrees of freedom, per
# number or per token, SGD and AdamW 3e-3, 1500 steps, TinyStories, 4 layers, C = 1.
from _common import PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--method", "ed", "--emb-scale", "4", "--wsd", "0.2",
           "--local-scale", "1", "--sim-cv", "ckpt/sim/cv_l4.json", "--sim-noise", "1", "--gamma", "0.98",
           "--data-dir", TINYSTORIES, "--steps", "1500"]
OPTIONS = {
    ".tail": {"values": ["3elem", "3token", "2.5token"],
              "flags": {"3elem": ["--sim-tail", "3,elem"], "3token": ["--sim-tail", "3,token"],
                        "2.5token": ["--sim-tail", "2.5,token"]}},
    ".opt": {"values": ["sgd", "adamw"], "flags": {"sgd": [], "adamw": ["--opt", "adamw", "--adam-lr", "3e-3"]}},
}
METRIC = "val_loss"
GOAL = "minimize"
