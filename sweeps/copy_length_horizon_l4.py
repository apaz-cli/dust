# At T = 2048 the recipe copies far worse than noise-free Dust (copy_length_l4, 3 seeds). Our reading: at gamma = 1
# the later-token leakage grows with the window (up to T/2 readers per key). Tests: a shorter credit horizon that
# still reaches the copy source (gamma 0.999: 0.999^1024 = 0.36; 0.9995: 0.60), and 4x the local draws at gamma 1
# (if the lower half's keys and values are the noise source, that should close much of the gap). Seeds 42 and 7.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--data-dir", "data/tsrepeat2048", "--batch-seqs", "2",
           "--cache-gamma", "0.98"] + LEAKFREE_CV
OPTIONS = {
    ".arm": {"values": ["g999", "g9995", "g1_local4"],
             "flags": {"g999": ["--gamma", "0.999", "--local-scale", "1"],
                       "g9995": ["--gamma", "0.9995", "--local-scale", "1"],
                       "g1_local4": ["--gamma", "1", "--local-scale", "4"]}},
    ".seed": {"values": [42, 7], "flags": "--seed"},
}
METRIC = "val_second_half"
GOAL = "minimize"
