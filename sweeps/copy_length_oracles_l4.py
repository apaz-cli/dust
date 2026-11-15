# Which sites' noise stops Dust copying from 1024 tokens back? Neither more key/value draws nor a shorter credit
# horizon helped (copy_length_horizon_l4). Oracle runs at T = 2048: the recipe (gamma 1 in the self-decoder, 0.98 on
# the cache), one group of sites given its noise-free update (derivatives as a measuring stick). Seed 42; the recipe
# alone reached 0.208 on the copied half (copy_length_l4).
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tsrepeat2048", "--batch-seqs", "2",
           "--gamma", "1", "--cache-gamma", "0.98", "--local-scale", "1"] + LEAKFREE_CV
OPTIONS = {
    ".group": {
        "values": ["selfattn", "cache", "selfwriters", "selfhidden", "embedding", "upperhalf"],
        "flags": {
            "selfattn": ["--oracle-re", r"h\.[01]\.attn\.(c_q|c_k|c_v|ve_gate)|value_embeds"],
            "cache": ["--oracle-re", r"cache_[kv]"],
            "selfwriters": ["--oracle-re", r"h\.[01]\.(attn|mlp)\.c_proj"],
            "selfhidden": ["--oracle-re", r"h\.[01]\.mlp\.c_fc"],
            "embedding": ["--oracle-re", r"transformer\.wte"],
            "upperhalf": ["--oracle-re", r"h\.[23]\."],
        },
    },
}
METRIC = "val_second_half"
GOAL = "minimize"
