# Noise of the local key/value estimators, stock vs leak-free scoring (bench_separate.py), on the
# backprop-trained 4-layer model, exact own-token attention-output targets (measuring stick only).
# Logged keys: <kind>/<site>/N<draws>/{stock,separate,cut}; cut = how many times less noise.
from _common import BP_COPY, BP_L4, COPY_TASK, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

# Also the cache on the backprop-trained copy-task model, where copying goes through the cache.
COMMAND = [PY, "yoco/bench_separate.py"]
OPTIONS = {".part": {"values": ["self", "cache", "cache_copy"],
                     "flags": {"self": [BP_L4, "1024", "self", TINYSTORIES], "cache": [BP_L4, "1024", "cache", TINYSTORIES],
                               "cache_copy": [BP_COPY, "1024", "cache", COPY_TASK]}}}
