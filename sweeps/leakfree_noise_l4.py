# Noise of the local key/value estimators, stock vs leak-free scoring (bench_separate.py), on the
# backprop-trained 4-layer model, exact own-token attention-output targets (measuring stick only).
# Logged keys: <kind>/<site>/N<draws>/{stock,separate,cut}; cut = how many times less noise.
from _common import BP_L4, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/bench_separate.py", BP_L4, "1024"]
OPTIONS = {".part": {"values": ["self", "cache"], "flags": {"self": ["self", TINYSTORIES], "cache": ["cache", TINYSTORIES]}}}
