# MoE follow-ups, 4-layer TinyStories, 1500 steps, seed 42 (MoE under Dust note).
#  - dense_mlp_oracle_adam: the missing control for "MoE with every MLP weight noise-free beats dense under Adam":
#    dense Dust with AdamW and its MLP weights given their noise-free update (derivatives as a measuring stick).
#  - E4k2 / E8k4: more tokens per expert (each expert sees k/E of the tokens: 1/2 instead of 1/4), same active size as
#    dense (k*h + shared = 1024), under backprop with AdamW and under Dust with AdamW and with SGD.
from _common import LEAKFREE_CV, TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

DUST = LEAKFREE_CV + ["--local-scale", "1"]
COMMAND = command(TINYSTORIES) + ["--steps", "1500", "--seed", "42"]
ADAM = ["--opt", "adamw", "--adam-lr", "3e-3"]
OPTIONS = {
    ".arm": {
        "values": ["dense_mlp_oracle_adam",
                   "E4k2_bp_adamw", "E4k2_dust_adamw", "E4k2_dust_sgd",
                   "E8k4_bp_adamw", "E8k4_dust_adamw", "E8k4_dust_sgd"],
        "flags": {
            "dense_mlp_oracle_adam": DUST + ADAM + ["--oracle-re", r"\.mlp\."],
            "E4k2_bp_adamw": ["--method", "bp"] + ADAM + ["--moe", "4,2,384,256"],
            "E4k2_dust_adamw": DUST + ADAM + ["--moe", "4,2,384,256"],
            "E4k2_dust_sgd": DUST + ["--lr", "0.25", "--moe", "4,2,384,256"],
            "E8k4_bp_adamw": ["--method", "bp"] + ADAM + ["--moe", "8,4,192,256"],
            "E8k4_dust_adamw": DUST + ADAM + ["--moe", "8,4,192,256"],
            "E8k4_dust_sgd": DUST + ["--lr", "0.25", "--moe", "8,4,192,256"],
        },
    },
}
METRIC = "val_loss"
GOAL = "minimize"
