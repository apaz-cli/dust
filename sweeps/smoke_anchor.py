# Debug: per-site error norms for the first steps, recipe with and without --anchor (4 and 8 layers).
from _common import L8, LEAKFREE_CV, PY, TINYSTORIES, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = ["env", "DUST_DEBUG=1", PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--steps", "6",
           "--eval-every", "100", "--data-dir", TINYSTORIES] + LEAKFREE_CV
OPTIONS = {".arm": {"values": ["l4_plain", "l4_a1", "l8_plain", "l8_a2"],
                    "flags": {"l4_plain": ["--local-scale", "1"], "l4_a1": ["--local-scale", "1", "--anchor", "1"],
                              "l8_plain": L8 + ["--local-scale", "4"], "l8_a2": L8 + ["--local-scale", "4", "--anchor", "2"]}}}
