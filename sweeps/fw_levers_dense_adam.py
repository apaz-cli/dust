# The dense AdamW reference for levers_moe_adam (fw_dust_moe_scale), which reached 4.281: dense with 4x writer and
# 4x MLP-input draws under AdamW 3e-3, 64 sequences. Only the SGD version was run (levers_dense_sgd, 4.347).
from fw_dust_moe_scale import *  # noqa: F401,F403
from fw_dust_moe_scale import ADAM, S, W4H4

OPTIONS = {".arm": {"values": ["levers_dense_adam"], "flags": {"levers_dense_adam": S(64) + W4H4 + ADAM}}}
