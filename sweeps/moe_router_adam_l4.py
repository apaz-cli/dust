# Does Adam break Dust MoE through the router? With AdamW the experts' load grows uneven (moe_load_max) and MoE
# stalls, while noise-free Dust with AdamW benefits from MoE. Current recipe, AdamW 3e-3, MoE, 4-layer TinyStories,
# 1500 steps, seed 42. router_frozen: router lr 0 (the routing stays at its random start, balanced by the bias).
from _common import LEAKFREE_CV, TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

COMMAND = command(TINYSTORIES) + LEAKFREE_CV + ["--local-scale", "1", "--steps", "1500", "--opt", "adamw",
                                                "--adam-lr", "3e-3", "--moe", "8,2,384,256"]
OPTIONS = {
    ".arm": {"values": ["router_lr0.1", "router_frozen", "router_oracle", "balance_fast", "router_lr0.1_balance_fast"],
             "flags": {"router_lr0.1": ["--router-lr-mult", "0.1"],
                       "router_frozen": ["--router-lr-mult", "0"],
                       "router_oracle": ["--oracle-re", r"mlp\.router"],
                       "balance_fast": ["--moe-balance-rate", "1e-2"],
                       "router_lr0.1_balance_fast": ["--router-lr-mult", "0.1", "--moe-balance-rate", "1e-2"]}},
}
METRIC = "val_loss"
GOAL = "minimize"
