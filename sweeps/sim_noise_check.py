# Noise simulator, checks and first uses (C from sim_noise_calib: about 1 at 4 layers, 0.6 at 8). Checks against real
# Dust: the copy task at 4 layers (recipe: copied half 0.078, readout_l4) and at T = 2048 (recipe: 0.208 to 0.423,
# copy_length_l4). Use: self-decoder depth under Dust-sized noise at 8 layers (1, 2, 4 layers; real-Dust runs in
# self_depth_dust are slow). Noise-free Dust has gamma 1 everywhere here (the recipe keeps 0.98 on the cache).
from _common import COPY_TASK, L8, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--method", "ed", "--emb-scale", "4", "--wsd", "0.2"]
L4 = ["--local-scale", "1", "--sim-cv", "ckpt/sim/cv_l4.json", "--sim-noise", "1"]
L8S = L8 + ["--local-scale", "4", "--clip", "2", "--sim-cv", "ckpt/sim/cv_l8.json", "--sim-noise", "0.6",
            "--gamma", "0.98", "--data-dir", TINYSTORIES]
OPTIONS = {".arm": {"values": ["copy512", "copy2048", "l8_n1", "l8_n2", "l8_n4"],
                    "flags": {"copy512": L4 + ["--gamma", "1", "--data-dir", COPY_TASK],
                              "copy2048": L4 + ["--gamma", "1", "--data-dir", "data/tsrepeat2048", "--batch-seqs", "2"],
                              "l8_n1": L8S + ["--n-self", "1"],
                              "l8_n2": L8S + ["--n-self", "2"],
                              "l8_n4": L8S}}}
METRIC = "val_loss"
GOAL = "minimize"
