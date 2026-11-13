# Real Dust with AdamW breaks MoE (moe_adamw_l4), while noise-free Dust with AdamW benefits from it. Does a
# noise-tolerant Adam fix it? Slower second-moment averaging (beta2 0.999), a larger epsilon (smaller steps where the
# estimate's running size is small), and a higher learning rate. Current recipe, 4-layer TinyStories, 1500 steps.
from _common import LEAKFREE_CV, TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

COMMAND = command(TINYSTORIES) + LEAKFREE_CV + ["--local-scale", "1", "--steps", "1500", "--opt", "adamw"]
OPTIONS = {
    ".adam": {"values": ["lr1e-2", "beta2_999", "eps1e-6", "eps1e-4"],
              "flags": {"lr1e-2": ["--adam-lr", "1e-2"],
                        "beta2_999": ["--adam-lr", "3e-3", "--adam-beta2", "0.999"],
                        "eps1e-6": ["--adam-lr", "3e-3", "--adam-eps", "1e-6"],
                        "eps1e-4": ["--adam-lr", "3e-3", "--adam-eps", "1e-4"]}},
    ".mlp": {"values": ["dense", "moe"], "flags": {"dense": [], "moe": ["--moe", "8,2,384,256"]}},
}
METRIC = "val_loss"
GOAL = "minimize"
