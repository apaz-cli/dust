# Diagnosing NaN at step 1 of compiled dense runs on the B300s (torch 2.11, CUDA 13); fine on torch 2.10 locally.
# --nan-debug runs one batch through the eager and the compiled model forward (full pass with recording, and
# truncated reruns from each block) and prints NaN counts, then exits. Backends: inductor (the default), aot_eager
# (no code generation), and a real 3-step run for comparison. Second round (here): 3 real steps with the
# per-site error norms printed (DUST_DEBUG), compiled, uncompiled, and compiled without code generation. Dense recipe, 32-sequence microbatch.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = ["env", "DUST_DEBUG=1", PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tinystories_long", "--local-scale", "1",
           "--microbatch-size", "32", "--batch-seqs", "64", "--opt", "adamw", "--adam-lr", "3e-3", "--steps", "3",
           "--eval-every", "1"] + LEAKFREE_CV
OPTIONS = {".arm": {"values": ["train3", "train3_nocompile", "train3_aot_eager"],
                    "flags": {"train3": [], "train3_nocompile": ["--no-compile"],
                              "train3_aot_eager": ["--compile-backend", "aot_eager"]}}}
METRIC = "val_loss"
GOAL = "minimize"
