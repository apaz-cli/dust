# Restart of the fw_round2 arms that were far from done on 18 November, with the compiled grouped MoE forward
# (--moe-grouped, now the default: about 2x faster per step). Same flags as fw_round2 otherwise; the finished and nearly finished arms were not rerun.
# The 8-layer AdamW arms (best8_moe_all_adam, place8_*) wait for a fix of the non-finite estimates that stopped both
# 8-layer dense AdamW runs near step 1000.
from fw_round2 import *  # noqa: F401,F403
from fw_round2 import OPTIONS as _O

OPTIONS = {".arm": {"values": ["best8_moe_all_sgd", "levers128_moe_sgd", "levers128_moe_adam", "writers8_moe_sgd", "writers8_moe_adam"], "flags": _O[".arm"]["flags"]}}
