# Step cost of the cross-token readout (bench_readout_time.py): residual-stream draws with and without it,
# alternated in one process, on the backprop-trained 4-layer model.
from _common import BP_L4, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/bench_readout_time.py", BP_L4, TINYSTORIES, "10"]
OPTIONS = {".rep": {"values": [1], "flags": {1: []}}}
