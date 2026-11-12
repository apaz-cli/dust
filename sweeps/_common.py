"""Shared pieces for the DustYOCO sweeps (campaign DustYOCO, local worker only).

Portable across workers: runs use `python` from the env mlsweep builds out of requirements.txt, and the data
and checkpoints they need ship with the upload (paths relative to the project root). The 17 GB of pre-mlsweep
runs and unused data stay out of the upload.
"""
PY = "python"
TINYSTORIES = "data/tinystories"     # 24k rows of 513 tokens, shared validation set
COPY_TASK = "data/tsrepeat"          # 256 story tokens, then the same 256 again
MQAR = "data/mqar8_k64v64"           # associative recall, 8 pairs per block, 64 keys, 64 values
UPLOAD_EXCLUDE = ["/runs", "/swarm_run", "/outputs", "/data/mqar128", "/data/mqar32", "/data/mqar8"]

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

# Backprop-trained checkpoints from the headline reruns, copied into the project so they ship with the upload:
# ckpt/bp_l4 <- headline_tinystories_l4_20261112_0303_d454/headline_tinystories_l4_armbackprop_seed42
# ckpt/bp_copy_l4 <- headline_copy_l4_20261112_0303_5f7c/headline_copy_l4_armbackprop_seed42
BP_L4 = "ckpt/bp_l4"
BP_COPY = "ckpt/bp_copy_l4"
