# Rerun of the runs of sweeps/lr_dust_l4.py that the old divergence rule stopped (a single step with training loss
# above 10; now 5 in a row). Same sweep, filtered to those runs, plus a "rerun" dimension so the names do not clash:
#   mlsweep run sweeps/rerun/lr_dust_l4.py --experiment <original experiment> --append
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_orig = {}
exec((Path(__file__).resolve().parents[1] / "lr_dust_l4.py").read_text(), _orig)
COMMAND, METRIC, GOAL, UPLOAD_EXCLUDE = _orig["COMMAND"], _orig["METRIC"], _orig["GOAL"], _orig["UPLOAD_EXCLUDE"]
OPTIONS = dict(_orig["OPTIONS"])
OPTIONS[".rerun"] = {"values": [1], "flags": {1: []}}
KEEP = [{'mlp': 'dense', 'lr': 0.5, 'seed': 42}, {'mlp': 'moe_m4', 'lr': 0.35, 'seed': 42}, {'mlp': 'moe_m4', 'lr': 0.5, 'seed': 42}]


def EXCLUDE(combo):
    return not any(all(combo.get(k) == v for k, v in want.items()) for want in KEEP)
