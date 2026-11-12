# Oracle runs: the best recipe (BEST, local x4) with one group of sites given its noise-free update (derivatives
# as a measuring stick), to measure what that group's draw noise costs. 4-layer TinyStories, seed 42.
from _common import BEST, TINYSTORIES, UPLOAD_EXCLUDE, command  # noqa: F401

COMMAND = command(TINYSTORIES) + BEST + ["--local-scale", "4"]
OPTIONS = {
    ".group": {
        "values": ["selfattn", "cache", "selfwriters", "selfhidden", "embedding", "upperhalf", "head"],
        "flags": {
            "selfattn": ["--oracle-re", r"h\.[01]\.attn\.(c_q|c_k|c_v|ve_gate)|value_embeds"],
            "cache": ["--oracle-re", r"cache_[kv]"],
            "selfwriters": ["--oracle-re", r"h\.[01]\.(attn|mlp)\.c_proj"],
            "selfhidden": ["--oracle-re", r"h\.[01]\.mlp\.c_fc"],
            "embedding": ["--oracle-re", r"transformer\.wte"],
            "upperhalf": ["--oracle-re", r"h\.[23]\."],
            "head": ["--oracle-re", r"lm_head"],
        },
    },
}
METRIC = "val_loss"
GOAL = "minimize"
