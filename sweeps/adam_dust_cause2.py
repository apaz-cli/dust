# Second part of adam_dust_cause: which of the two, the control variate or live units, makes AdamW hurt real Dust.
# Why does AdamW hurt real Dust (2.643 against 2.556 with SGD at 1500 steps, moe_adamw_l4 / lr_dust_l4) when the noise
# simulator, with Gaussian, heavy-tailed or floored noise of Dust's size, says AdamW should help? One candidate the
# simulator lacks: the control variate's memory carries errors from step to step, so its mistakes are correlated over
# time, which Adam's per-coordinate scaling could amplify. Real Dust, recipe, AdamW 3e-3, 1500 steps, TinyStories,
# 4 layers: without the control variate (and live units, as in the recipe without them), and with a faster memory.
# adam_dust_cause2 runs the control variate and live units separately: live units leave inactive MLP units with
# exactly zero updates, and Adam then scales a unit's first nonzero estimate by a tiny running average.
from _common import LEAKFREE, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", TINYSTORIES, "--steps", "1500",
           "--local-scale", "1"] + LEAKFREE
OPTIONS = {".arm": {"values": ["cv_only_adamw", "live_only_adamw"],
                    "flags": {"nocv_adamw": ["--opt", "adamw", "--adam-lr", "3e-3"],
                              "nocv_sgd": [],
                              "cv_beta05_adamw": ["--live-fc", "--cv", "--cv-beta", "0.5", "--opt", "adamw",
                                                  "--adam-lr", "3e-3"],
                              "cv_only_adamw": ["--cv", "--opt", "adamw", "--adam-lr", "3e-3"],
                              "live_only_adamw": ["--live-fc", "--opt", "adamw", "--adam-lr", "3e-3"]}}}
METRIC = "val_loss"
GOAL = "minimize"
