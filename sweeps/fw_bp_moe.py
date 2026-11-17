# fineweb-edu, step 1: does MoE beat dense under backprop on this data, and does the gain grow with more experts at a
# fixed number of tokens per expert? Width 512, 4 layers (2-layer self-decoder), 1500 steps, AdamW. Dense MLP hidden
# 2048; MoE: 2 active experts of 768 plus a shared expert of 512 (the same 2048 active). 8 experts at 64 sequences,
# 16 at 128, 32 at 256 (about 8k tokens per expert); dense at each batch; AdamW 1e-3 and 3e-3 at 64 sequences.
from _common import PY, UPLOAD_EXCLUDE  # noqa: F401

FW = ["--data-dir", "data/fineweb_edu", "--n-layer", "4", "--n-embd", "512", "--steps", "1500"]
COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--method", "bp", "--opt", "adamw"] + FW
OPTIONS = {".arm": {"values": ["dense_s64_lr3", "moe8_s64_lr3", "dense_s64_lr1", "moe8_s64_lr1",
                               "dense_s128", "moe16_s128", "dense_s256", "moe32_s256"],
                    "flags": {"dense_s64_lr3": ["--batch-seqs", "64", "--adam-lr", "3e-3"],
                              "moe8_s64_lr3": ["--batch-seqs", "64", "--adam-lr", "3e-3", "--moe", "8,2,768,512"],
                              "dense_s64_lr1": ["--batch-seqs", "64", "--adam-lr", "1e-3"],
                              "moe8_s64_lr1": ["--batch-seqs", "64", "--adam-lr", "1e-3", "--moe", "8,2,768,512"],
                              "dense_s128": ["--batch-seqs", "128", "--adam-lr", "3e-3"],
                              "moe16_s128": ["--batch-seqs", "128", "--adam-lr", "3e-3", "--moe", "16,2,768,512"],
                              "dense_s256": ["--batch-seqs", "256", "--adam-lr", "3e-3"],
                              "moe32_s256": ["--batch-seqs", "256", "--adam-lr", "3e-3", "--moe", "32,2,768,512"]}}}
METRIC = "val_loss"
GOAL = "minimize"
