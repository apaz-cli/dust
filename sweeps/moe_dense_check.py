# Correctness of the fixed-shape MoE forward (--moe-dense: every expert on every token, the routed ones kept, so the
# model forward can compile): 3 steps against the sparse path, uncompiled, same batch; losses and gradient norms
# per model part should match closely. Then the same with compile on.
from _common import LEAKFREE_CV, PY, UPLOAD_EXCLUDE as _EX  # noqa: F401

UPLOAD_EXCLUDE = _EX + ["/data/fineweb_edu"]  # not needed here (and being written)

COMMAND = [PY, "yoco/train.py", "--arch", "yoco", "--seed", "42", "--data-dir", "data/tinystories_long", "--local-scale", "1",
           "--microbatch-size", "32", "--batch-seqs", "32", "--opt", "adamw", "--adam-lr", "3e-3", "--steps", "3",
           "--eval-every", "1", "--moe", "8,2,384,256", "--moe-fix", "--router-cap", "2", "--router-wd", "10"] + LEAKFREE_CV
OPTIONS = {".arm": {"values": ["sparse_eager", "dense_eager", "dense_compiled"],
                    "flags": {"sparse_eager": ["--no-compile"], "dense_eager": ["--moe-dense", "--no-compile"],
                              "dense_compiled": ["--moe-dense"]}}}
METRIC = "val_loss"
GOAL = "minimize"
