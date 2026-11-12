# Exact leakage and target quality per site (measure_bias.py; derivatives used as a measuring stick only) on the
# backprop-trained 4-layer model from the headline rerun. "own" = plain Dust's target; numbers = credit discount
# gamma for later tokens. Logged keys: <metric>/<site>, e.g. kappa/h.0.attn.c_k, wcos_g/cache_k.
from _common import BP_L4, PY, UPLOAD_EXCLUDE  # noqa: F401

COMMAND = [PY, "yoco/measure_bias.py", "--run", BP_L4, "--sequences", "2"]
OPTIONS = {
    ".gamma": {
        "values": ["own", "0.9", "0.95", "0.98", "0.99", "0.995", "0.999"],
        "flags": {"own": [], **{g: ["--gamma", g] for g in ["0.9", "0.95", "0.98", "0.99", "0.995", "0.999"]}},
    },
}
