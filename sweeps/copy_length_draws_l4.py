# At T = 2048 Dust's copy noise is spread over the whole model (copy_length_oracles_l4). Does cutting noise everywhere
# close the gap to noise-free Dust (0.103)? The recipe with 2x and 4x the draws at every site (writers, MLP inputs,
# embedding, attention internals), and 2x plus the cross-token readout. Seed 42; the recipe alone: 0.208.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tsrepeat2048", "--batch-seqs", "2",
           "--gamma", "1", "--cache-gamma", "0.98"] + LEAKFREE_CV
OPTIONS = {".arm": {"values": ["x2", "x4", "x2_readout"],
                    "flags": {"x2": ["--direct-scale", "2", "--emb-scale", "8", "--local-scale", "2"],
                              "x4": ["--direct-scale", "4", "--emb-scale", "16", "--local-scale", "4"],
                              "x2_readout": ["--direct-scale", "2", "--emb-scale", "8", "--local-scale", "2",
                                             "--readout"]}}}
METRIC = "val_second_half"
GOAL = "minimize"
