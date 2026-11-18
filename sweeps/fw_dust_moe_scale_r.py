# Restart of the fw_dust_moe_scale arms that were far from done on 18 November, with the compiled grouped MoE forward
# (--moe-grouped, now the default: about 2x faster per step). Same flags as fw_dust_moe_scale otherwise; the finished and nearly finished arms were not rerun.
from fw_dust_moe_scale import *  # noqa: F401,F403
from fw_dust_moe_scale import OPTIONS as _O

OPTIONS = {".arm": {"values": ["scale_moe16_sgd", "scale_moe16_adam", "scale_moe32_sgd", "scale_moe32_adam"], "flags": _O[".arm"]["flags"]}}
