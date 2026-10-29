# YOCO × Dust experiments

Derivative-free (zeroth-order) training of YOCO models with Dust, and tools that separate Dust's gradient error
into bias, leakage, draw noise and bf16 rounding. Results are logged in `FINDINGS.md`; the write-up is
`~/YOCO_Dust_2_Results.pdf`. Run everything on one GPU: `CUDA_VISIBLE_DEVICES=0`.

Data: `data/tinystories` is a symlink to `~/git/dust/data/tinystories` (512-token rows).

## Models and training

| file | what |
|---|---|
| `model_yoco.py` | YOCO: windowed self-decoder, one shared KV cache (`cache_k`, `cache_v`), cross-decoder that reads only the cache. `build('gpt' or 'yoco', ...)` |
| `dustx.py` | `DustX(DUST)`: stock Dust plus switches (see below) |
| `expected.py` | noise-free Dust: backprop of the update Dust targets in expectation (measurement tool) |
| `train.py` | one entry point for `--method dust`, `bp` (backprop) or `ed` (noise-free Dust) |

Derivative-free `train.py` switches (the recommended YOCO recipe is the first three):

```
--fp32             Dust forwards in fp32 activations with TF32 matmuls (removes bf16 rounding from rewards)
--local-scale 4    4x ZO draws for keys, values, gates, value embeddings and the YOCO cache
--hub-scale 4      4x ZO draws for the attention-output (o@) errors those are scored against
--direct-scale K   Kx writer / MLP draws          --emb-scale K   Kx embedding draws
--head-div K       1/K of the ZO head draws       --flat-alloc    same draws in every block
--gamma G          key/value credit discount      --res-gamma G   discounted credit for blocks 0-1 (hurt)
--credit attn      attention-matched k/v credit (hurt in training)
```

Diagnostic switches that use derivatives (not methods): `--local vjp`, `--head exact`, `--handoff`, `--oracle GROUPS`.

Best derivative-free YOCO run (L4-D256, 3000 steps): val 2.842 vs stock 3.131:

```
python yoco/train.py --arch yoco --method dust --fp32 --local-scale 4 --hub-scale 4 --output runs/best
```

## Measurement tools (autograd only as a ruler)

| file | measures |
|---|---|
| `check_exact.py`, `check_expected.py` | exact parts and the noise-free Dust backward against autograd |
| `measure_bias.py` | per site: cosine of Dust's expected update to the true gradient, and exact leakage kappa |
| `error_budget.py` | per site: bias², draw noise per draw, bf16 rounding per draw |
| `bench_site_precision.py`, `bench_hub_precision.py` | estimate quality in bf16 vs fp32 |
| `bench_local_zo.py` | key / value / cache estimator designs and credit kernels, incl. the full pipeline |
| `bench_orthogonal.py` | Gaussian vs orthogonal probe directions; leakage sets the floor |
| `bench_res_credit.py` | residual discounted credit, total error vs the true gradient |
| `context_use.py`, `position_curves.py` | validation loss by token position |
| `bench_local.py` | per-step cost of each estimator part |
| `summarize_ed.py` | seed means for backprop and noise-free arms |

Queues `queue*.sh` record the exact arms that were run.
