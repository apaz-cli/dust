# Why does AdamW hurt real Dust (2.643 against 2.556 with SGD at 1500 steps, moe_adamw_l4 / lr_dust_l4) when the noise
# simulator, with Gaussian, heavy-tailed or floored noise of Dust's size, says AdamW should help? One candidate the
# simulator lacks: the control variate's memory carries errors from step to step, so its mistakes are correlated over
# time, which Adam's per-coordinate scaling could amplify. Real Dust, recipe, AdamW 3e-3, 1500 steps, TinyStories,
# 4 layers: without the control variate (and live units, as in the recipe without them), and with a faster memory.
from _common import LEAKFREE, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", TINYSTORIES, "--steps", "1500",
           "--local-scale", "1"] + LEAKFREE
OPTIONS = {".arm": {"values": ["nocv_adamw", "nocv_sgd", "cv_beta05_adamw"],
                    "flags": {"nocv_adamw": ["--opt", "adamw", "--adam-lr", "3e-3"],
                              "nocv_sgd": [],
                              "cv_beta05_adamw": ["--live-fc", "--cv", "--cv-beta", "0.5", "--opt", "adamw",
                                                  "--adam-lr", "3e-3"]}}}
METRIC = "val_loss"
GOAL = "minimize"
