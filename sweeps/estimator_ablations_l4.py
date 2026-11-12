# Changes to the estimator, each against night 2's best recipe (BEST, local x4), 4-layer TinyStories, seed 42.
# Rows for the Estimators notes (hub chaining, leak-free scoring, live units + control variate, dead ends).
# Headline arms (best, leakfree, leakfree_cv) are in headline_tinystories_l4.
from _common import BEST, TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

NO_CHAIN = [f for f in BEST if f not in ("--chain", "256", "--chain-own", "--chain-hubs-only")]
NO_WSD = [f for f in BEST if f not in ("--wsd", "0.2")]
L4X = ["--local-scale", "4"]
COMMAND = command(TINYSTORIES)
OPTIONS = {
    ".arm": {
        "values": ["hub_draws", "no_wsd", "hub_draws_no_wsd", "leakfree_self", "leakfree_self_l1", "stock_local_l1",
                   "live_cv_no_leakfree", "attn_credit", "residual_credit", "pooling", "late_draws", "batch16",
                   "nvfp4_master", "nvfp4_sr", "cache_gamma_995", "linear_values"],
        "flags": {
            "hub_draws": NO_CHAIN + L4X,                         # attention-output errors from full reruns
            "no_wsd": NO_WSD + L4X,
            "hub_draws_no_wsd": [f for f in NO_CHAIN if f not in ("--wsd", "0.2")] + L4X,
            "leakfree_self": BEST + L4X + ["--local-separate"],
            "leakfree_self_l1": BEST + ["--local-scale", "1", "--local-separate"],
            "stock_local_l1": BEST + ["--local-scale", "1"],     # control for leakfree_self_l1
            "live_cv_no_leakfree": BEST + L4X + ["--live-fc", "--cv"],
            "attn_credit": BEST + L4X + ["--credit", "attn"],
            "residual_credit": BEST + L4X + ["--res-gamma", "0.5"],
            "pooling": BEST + L4X + ["--pool"],
            "late_draws": BEST + L4X + ["--draw-ramp"],
            "batch16": BEST + L4X + ["--batch-seqs", "16", "--steps", "1500", "--lr", "0.35"],
            "nvfp4_master": BEST + L4X + ["--quant", "master"],
            "nvfp4_sr": BEST + L4X + ["--quant", "sr"],
            "cache_gamma_995": BEST + L4X + ["--cache-gamma", "0.995"],
            "linear_values": BEST + L4X + ["--linear-v"],        # gray zone: uses attention's linearity in values
        },
    },
    ".seed": {"values": [42], "flags": "--seed"},
}
METRIC = "val_loss"
GOAL = "minimize"
