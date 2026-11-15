# At T = 2048 neither more draws (copy_length_draws_l4) nor any single oracle group (copy_length_oracles_l4) closed the gap
# between the recipe (copied half 0.208) and noise-free Dust (0.103). The one difference left between them: the recipe
# keeps gamma 0.98 on the cache, which gives no cache credit at 1024 tokens back; noise-free Dust had gamma 1 everywhere.
# The recipe with gamma 1 on the cache too, with and without twice the draws. Seed 42.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tsrepeat2048", "--batch-seqs", "2",
           "--gamma", "1", "--cache-gamma", "1", "--local-scale", "1"] + LEAKFREE_CV
OPTIONS = {".arm": {"values": ["x1", "x2"],
                    "flags": {"x1": [], "x2": ["--direct-scale", "2", "--emb-scale", "8", "--local-scale", "2"]}}}
METRIC = "val_second_half"
GOAL = "minimize"
