# Fair reference for moe_next_round w4_s32 (MoE with 4x writer draws, 2.429 against 2.580 without): dense with 4x writer
# draws at 32 sequences (dense with the recipe's draws: 2.286). Same flags as moe_next_round.
from moe_next_round import COMMAND, GOAL, METRIC, UPLOAD_EXCLUDE  # noqa: F401

OPTIONS = {".arm": {"values": ["dense_w4_s32"],
                    "flags": {"dense_w4_s32": ["--batch-seqs", "32", "--steps", "1500",
                                               "--alloc", '{"writers": [152, 128, 64, 40]}']}}}
