# Benchmark (Plan item 2): the symmetric readout. Against the true gradient, on the backprop-trained 4-layer models:
# plain Dust, the readout's credit term, the leakage-cancelling term, and both, with exact and chained targets
# (bench_readout.py sym).
from _common import BP_COPY, BP_L4, COPY_TASK, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/bench_readout.py"]
OPTIONS = {".model": {"values": ["tinystories", "copy"],
                      "flags": {"tinystories": [BP_L4, TINYSTORIES, "3", "sym"], "copy": [BP_COPY, COPY_TASK, "3", "sym"]}}}
