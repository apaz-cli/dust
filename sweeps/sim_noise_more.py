# Noise simulator, round 2. (1) Key/value noise sized by the head width instead of the stream width (leak-free scoring
# estimates each key head by head): does it fix the too-pessimistic copy result at T = 2048 (real recipe 0.21 to
# 0.42), and keep T = 512 and TinyStories right? (2) AdamW under Dust-sized noise, 1500 steps: real Dust with AdamW
# 3e-3 reached 2.643 against 2.556 with SGD (moe_adamw_l4, lr_dust_l4); does the simulator show the same?
from _common import COPY_TASK, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--method", "ed", "--emb-scale", "4", "--wsd", "0.2",
           "--local-scale", "1", "--sim-cv", "ckpt/sim/cv_l4.json", "--sim-noise", "1"]
OPTIONS = {".arm": {"values": ["kvhead_copy2048", "kvhead_copy512", "kvhead_ts", "sgd1500", "adamw1500"],
                    "flags": {"kvhead_copy2048": ["--sim-kv-head", "--gamma", "1", "--data-dir", "data/tsrepeat2048",
                                                  "--batch-seqs", "2"],
                              "kvhead_copy512": ["--sim-kv-head", "--gamma", "1", "--data-dir", COPY_TASK],
                              "kvhead_ts": ["--sim-kv-head", "--gamma", "0.98", "--data-dir", TINYSTORIES],
                              "sgd1500": ["--gamma", "0.98", "--data-dir", TINYSTORIES, "--steps", "1500"],
                              "adamw1500": ["--gamma", "0.98", "--data-dir", TINYSTORIES, "--steps", "1500",
                                            "--opt", "adamw", "--adam-lr", "3e-3"]}}}
METRIC = "val_loss"
GOAL = "minimize"
