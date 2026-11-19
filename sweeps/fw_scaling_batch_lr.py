# Validation of [[Scaling theory]] (notes): learning rate against batch size, and Dust's critical batch size against
# backprop's. Dense, width 512, 4 layers, fineweb-edu, 1500 steps, the fw_dust_dense setup (local draws x4,
# microbatches of 16), all on the current code.
# - Dust SGD at 32, 64, 128, 256 sequences, the learning rate at 0.5, 1 and 2 times the linear rule
#   (0.25 at 64 sequences, the recipe's rate). Prediction: the best rate follows the linear rule, and loss keeps
#   improving with batch further than backprop's does.
# - Dust AdamW at the same batches, the rate at 0.5, 1, 2 times the square-root rule (3e-3 at 64, the best of
#   fw_dust_dense). Prediction: the best rate follows the square-root rule.
# - Backprop AdamW at the same batches and rates, for backprop's critical batch size.
# - Dust SGD with every draw count doubled (--direct-scale 2, local draws x8) at 64 and 128 sequences: total draws
#   against batch, at about the cost of the next batch size up.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

FW = ["--data-dir", "data/fineweb_edu", "--n-layer", "4", "--n-embd", "512", "--steps", "1500", "--save-every", "0"]
COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42"] + FW
DUST = ["--local-scale", "4", "--microbatch-size", "16"] + LEAKFREE_CV
BATCHES = [32, 64, 128, 256]
MULTS = {"x0.5": 0.5, "x1": 1.0, "x2": 2.0}

flags = {}
for b in BATCHES:
    for m, f in MULTS.items():
        sgd_lr = 0.25 * (b / 64) * f
        adam_lr = 3e-3 * (b / 64) ** 0.5 * f
        flags[f"dust_sgd_b{b}_{m}"] = DUST + ["--batch-seqs", str(b), "--lr", f"{sgd_lr:.4g}"]
        flags[f"dust_adam_b{b}_{m}"] = DUST + ["--batch-seqs", str(b), "--opt", "adamw", "--adam-lr", f"{adam_lr:.4g}"]
        flags[f"bp_adam_b{b}_{m}"] = ["--method", "bp", "--opt", "adamw", "--wsd", "0.2", "--batch-seqs", str(b),
                                      "--adam-lr", f"{adam_lr:.4g}"]
for b in (64, 128):
    flags[f"dust_sgd_b{b}_x1_draws2"] = ["--local-scale", "8", "--direct-scale", "2", "--microbatch-size", "16"] \
        + LEAKFREE_CV + ["--batch-seqs", str(b), "--lr", f"{0.25 * b / 64:.4g}"]

OPTIONS = {".arm": {"values": list(flags), "flags": flags}}
METRIC = "val_loss"
GOAL = "minimize"
