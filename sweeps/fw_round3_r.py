# Restart of the fw_round3 arms that were far from done on 18 November, with the compiled grouped MoE forward
# (--moe-grouped, now the default: a Triton grouped matmul, yoco/gmm.py; 2.4x faster per step on the RTX 5090). Same flags as fw_round3 otherwise; the finished and nearly finished arms were not rerun.
from fw_round3 import *  # noqa: F401,F403
from fw_round3 import OPTIONS as _O

OPTIONS = {".arm": {"values": ["moe16_h4", "moe8_h4", "moe8_h8"], "flags": _O[".arm"]["flags"]}}
