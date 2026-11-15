# The best8 arms of moe_dust_adamw, restarted: Dust counts readout warm-up steps per microbatch, so at 64 sequences in
# microbatches of 8 the warm-up of 100 was only about 12 optimizer steps; 800 here is 100 optimizer steps.
from moe_dust_adamw import COMMAND, GOAL, L8, LEAKFREE_CV, METRIC, MOE, PY, UPLOAD_EXCLUDE  # noqa: F401

BEST8 = L8 + ["--clip", "2", "--local-scale", "4", "--n-self", "2", "--readout", "--readout-warmup", "800",
              "--direct-scale", "2", "--emb-scale", "8", "--steps", "1500", "--batch-seqs", "64"]
OPTIONS = {".arm": {"values": ["best8_dense", "best8_moe_upper"],
                    "flags": {"best8_dense": BEST8,
                              "best8_moe_upper": BEST8 + MOE + ["--moe-layers", "2", "3", "4", "5", "6", "7"]}}}
