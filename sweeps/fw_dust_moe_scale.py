# fineweb-edu, step 3: the MoE scaling test under Dust with every fix, and all levers together. Width 512, 4 layers
# (2-layer self-decoder), 1500 steps, microbatches of 16. Backprop's MoE gain grows with experts at a fixed number of
# tokens per expert on this data (fw_bp_moe: 0.057, 0.093, 0.121 at 8, 16, 32 experts). Dense Dust preferred SGD here
# (fw_dust_dense), MoE needed AdamW on TinyStories, so MoE runs both. All Dust runs: 4x the writer draws (the experts'
# output weights come from the writer error, moe_next_round), and for MoE the fix and the router stabilizers.
# scale_*: 8 experts at 64 sequences, 16 at 128, 32 at 256 (2 active of 768 plus a shared 512), dense at each batch.
# levers_*: 64 sequences, experts in the lower half only, 4x writer and 4x MLP-input draws, per-expert control variate.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

FW = ["--data-dir", "data/fineweb_edu", "--n-layer", "4", "--n-embd", "512", "--steps", "1500"]
COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--local-scale", "4", "--microbatch-size", "16"] + FW + LEAKFREE_CV
W4 = ["--alloc", '{"writers": [152, 128, 64, 40]}']
W4H4 = ["--alloc", '{"writers": [152, 128, 64, 40], "hidden": [152, 128, 64, 40]}']
FIX = ["--moe-fix", "--router-cap", "2", "--router-wd", "10"]
ADAM = ["--opt", "adamw", "--adam-lr", "3e-3"]
S = lambda n: ["--batch-seqs", str(n)]
M = lambda e: ["--moe", f"{e},2,768,512"] + FIX
OPTIONS = {".arm": {"values": ["scale_moe8_sgd", "scale_moe8_adam", "scale_dense64_sgd", "scale_dense64_adam",
                               "scale_moe16_sgd", "scale_moe16_adam", "scale_dense128_sgd",
                               "scale_moe32_sgd", "scale_moe32_adam", "scale_dense256_sgd",
                               "levers_moe_sgd", "levers_moe_adam", "levers_dense_sgd"],
                    "flags": {"scale_moe8_sgd": S(64) + M(8) + W4, "scale_moe8_adam": S(64) + M(8) + W4 + ADAM,
                              "scale_dense64_sgd": S(64) + W4, "scale_dense64_adam": S(64) + W4 + ADAM,
                              "scale_moe16_sgd": S(128) + M(16) + W4, "scale_moe16_adam": S(128) + M(16) + W4 + ADAM,
                              "scale_dense128_sgd": S(128) + W4,
                              "scale_moe32_sgd": S(256) + M(32) + W4, "scale_moe32_adam": S(256) + M(32) + W4 + ADAM,
                              "scale_dense256_sgd": S(256) + W4,
                              "levers_moe_sgd": S(64) + M(8) + W4H4 + ["--moe-layers", "0", "1", "--moe-cv"],
                              "levers_moe_adam": S(64) + M(8) + W4H4 + ["--moe-layers", "0", "1", "--moe-cv"] + ADAM,
                              "levers_dense_sgd": S(64) + W4H4}}}
METRIC = "val_loss"
GOAL = "minimize"
