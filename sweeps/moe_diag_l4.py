# Why does MoE lose under Dust? 4-layer TinyStories, 1500 steps (as lr_dust_l4 / lr_bp_l4), learning rates from
# those sweeps (Dust 0.25, backprop SGD 0.2, AdamW 3e-3). MoE: 8 experts, top 2, hidden 384, shared expert 256.
#  - seed7_*: second seed at the best learning rate.
#  - noisefree_*: noise-free Dust (Dust's average, computed exactly): does Dust's target benefit from MoE?
#  - oracle_*: Dust MoE with one group made noise-free (derivatives as a measuring stick): which noise hurts?
from _common import LEAKFREE_CV, TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

MOE = ["--moe", "8,2,384,256"]
WIDE = ["--moe", "1,1,3072,256"]
DUST = LEAKFREE_CV + ["--local-scale", "1", "--lr", "0.25"]
COMMAND = command(TINYSTORIES) + ["--steps", "1500"]
OPTIONS = {
    ".arm": {
        "values": ["seed7_dust_dense", "seed7_dust_moe", "seed7_bp_sgd_dense", "seed7_bp_sgd_moe",
                   "seed7_bp_adamw_dense", "seed7_bp_adamw_moe",
                   "noisefree_dense", "noisefree_moe", "noisefree_wide",
                   "oracle_router", "oracle_experts", "oracle_shared", "oracle_all_mlp", "oracle_dense_mlp"],
        "flags": {
            "seed7_dust_dense": DUST + ["--seed", "7"],
            "seed7_dust_moe": DUST + MOE + ["--seed", "7"],
            "seed7_bp_sgd_dense": ["--method", "bp", "--lr", "0.2", "--seed", "7"],
            "seed7_bp_sgd_moe": ["--method", "bp", "--lr", "0.2", "--seed", "7"] + MOE,
            "seed7_bp_adamw_dense": ["--method", "bp", "--opt", "adamw", "--adam-lr", "3e-3", "--seed", "7"],
            "seed7_bp_adamw_moe": ["--method", "bp", "--opt", "adamw", "--adam-lr", "3e-3", "--seed", "7"] + MOE,
            "noisefree_dense": ["--method", "ed", "--gamma", "0.98", "--lr", "0.2"],
            "noisefree_moe": ["--method", "ed", "--gamma", "0.98", "--lr", "0.2"] + MOE,
            "noisefree_wide": ["--method", "ed", "--gamma", "0.98", "--lr", "0.2"] + WIDE,
            "oracle_router": DUST + MOE + ["--oracle-re", r"mlp\.router"],
            "oracle_experts": DUST + MOE + ["--oracle-re", r"mlp\.w_(fc|proj)"],
            "oracle_shared": DUST + MOE + ["--oracle-re", r"mlp\.c_(fc|proj)"],
            "oracle_all_mlp": DUST + MOE + ["--oracle-re", r"\.mlp\."],
            "oracle_dense_mlp": DUST + ["--oracle-re", r"\.mlp\."],
        },
    },
}
METRIC = "val_loss"
GOAL = "minimize"
