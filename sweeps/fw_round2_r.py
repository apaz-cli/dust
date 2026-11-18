# Restart of the fw_round2 arms that were far from done on 18 November, with the compiled grouped MoE forward
# (--moe-grouped, now the default: a Triton grouped matmul, yoco/gmm.py; 2.4x faster per step on the RTX 5090). Same flags as fw_round2 otherwise; the finished and nearly finished arms were not rerun.
# The 8-layer AdamW arms (best8_*_adam, place8_*, with their dense references) also get the fix for the non-finite
# estimates that stopped both 8-layer dense AdamW runs near step 1000: under fast16 the output projections now return
# fp32 (the first block's MLP output had grown to about 6e4, near fp16's limit; yoco/diag_overflow.py).
from fw_round2 import *  # noqa: F401,F403
from fw_round2 import OPTIONS as _O

OPTIONS = {".arm": {"values": ["best8_moe_all_sgd", "best8_moe_all_adam", "best8_moe_lower_adam", "best8_dense_adam", "levers128_moe_sgd", "levers128_moe_adam", "place8_all", "place8_lower", "place8_upper", "place8_dense", "writers8_moe_sgd", "writers8_moe_adam"], "flags": _O[".arm"]["flags"]}}
