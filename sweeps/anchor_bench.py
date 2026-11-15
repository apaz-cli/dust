# Benchmark (depth cost, [[Plan]]): anchored draws, which rerun only up to a nearby block and score the stream change
# there against an error estimate, against full truncated draws, on the backprop-trained models (bench_anchor.py).
from _common import BP_COPY, BP_L4, PY, TINYSTORIES, COPY_TASK, UPLOAD_EXCLUDE  # noqa: F401

BP_L8 = "ckpt/bp_l8"  # <- headline_tinystories_l8_20261112_0303_6261/headline_tinystories_l8_armbackprop_seed42
COMMAND = [PY, "yoco/bench_anchor.py"]
OPTIONS = {".model": {"values": ["l8", "l4", "l4copy"],
                      "flags": {"l8": [BP_L8, TINYSTORIES, "3"], "l4": [BP_L4, TINYSTORIES, "3"],
                                "l4copy": [BP_COPY, COPY_TASK, "3"]}}}
