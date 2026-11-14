# Benchmark: how accurate the attention-output targets must be for the cross-token readout to help: exact targets
# plus noise at a chosen cosine, and Dust hubs at more draws, at 256 writer draws (bench_readout.py quality).
from _common import BP_COPY, BP_L4, COPY_TASK, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/bench_readout.py"]
OPTIONS = {".model": {"values": ["tinystories", "copy"],
                      "flags": {"tinystories": [BP_L4, TINYSTORIES, "3", "quality"],
                                "copy": [BP_COPY, COPY_TASK, "3", "quality"]}}}
