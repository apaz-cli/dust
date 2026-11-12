"""Shared pieces for the DustYOCO sweeps (campaign DustYOCO, local worker only).

Runs use this repo's own venv (torch 2.10, what the code was tested with) and absolute data paths, since the
only worker is this machine. The 17 GB of pre-mlsweep runs and the data folder stay out of the upload.
"""
PY = "/home/apaz/git/dust_experiments/.venv/bin/python"
TINYSTORIES = "/home/apaz/git/dust/data/tinystories"          # 24k rows of 513 tokens, shared validation set
COPY_TASK = "/home/apaz/git/dust_experiments/data/tsrepeat"     # 256 story tokens, then the same 256 again
UPLOAD_EXCLUDE = ["/runs", "/swarm_run", "/data", "/outputs"]

L8 = ["--n-layer", "8", "--n-embd", "512"]

# Night 2's best derivative-free recipe: fp16 matmuls with fp32 rewards, draws rerun from the perturbed block,
# attention-output errors from cheap local probes, 4x local and embedding draws, WSD learning-rate decay.
BEST = ["--method", "dust", "--precision", "fast16", "--chain", "256", "--chain-own", "--chain-hubs-only",
        "--emb-scale", "4", "--wsd", "0.2"]
# Plus leak-free scoring of keys and values (each token's perturbation scored alone), self-decoder and cache.
LEAKFREE = BEST + ["--local-separate", "--cache-separate"]
# Plus night 1's two noise reductions: perturb only active MLP units, and the head-error control variate.
LEAKFREE_CV = LEAKFREE + ["--live-fc", "--cv"]
STOCK = ["--method", "dust"]


def command(data):
    return [PY, "yoco/train.py", "--arch", "yoco", "--data-dir", data]

EXP = "/home/apaz/.mlsweep/experiments"
# Checkpoints from the headline reruns, used by the measurement sweeps.
BP_L4 = f"{EXP}/headline_tinystories_l4_20261112_0303_d454/headline_tinystories_l4_armbackprop_seed42/artifacts"
BP_L8 = f"{EXP}/headline_tinystories_l8_20261112_0303_6261/headline_tinystories_l8_armbackprop_seed42/artifacts"
BP_COPY = f"{EXP}/headline_copy_l4_20261112_0303_5f7c/headline_copy_l4_armbackprop_seed42/artifacts"
