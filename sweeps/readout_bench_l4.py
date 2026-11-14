# Benchmark: cross-token readout (noise PR #46 adapted to plain Dust) vs plain Dust, against the true gradient,
# on the backprop-trained 4-layer TinyStories and copy-task models (bench_readout.py).
from _common import BP_COPY, BP_L4, COPY_TASK, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/bench_readout.py"]
OPTIONS = {".model": {"values": ["tinystories", "copy"],
                      "flags": {"tinystories": [BP_L4, TINYSTORIES, "3"], "copy": [BP_COPY, COPY_TASK, "3"]}}}
