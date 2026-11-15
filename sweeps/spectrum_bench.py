# Measurement (Plan item 5): are the per-token errors at each Dust site concentrated in a few directions? Share of
# held-out error energy in the top-r directions fit on other batches, on the backprop-trained models (bench_spectrum.py).
from _common import BP_COPY, BP_L4, COPY_TASK, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

BP_L8 = "ckpt/bp_l8"
COMMAND = [PY, "yoco/bench_spectrum.py"]
OPTIONS = {".model": {"values": ["l4", "l4copy", "l8"],
                      "flags": {"l4": [BP_L4, TINYSTORIES, "16"], "l4copy": [BP_COPY, COPY_TASK, "16"],
                                "l8": [BP_L8, TINYSTORIES, "16"]}}}
