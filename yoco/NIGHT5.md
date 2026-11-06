# Night 5 results (branch `kappa-moe`; every number also in FINDINGS.md, "Night 5" sections)

All runs: YOCO, TinyStories unless noted, 512 tokens, 8 sequences per step, 3000 steps, val loss, single seed unless
noted. "Best recipe" = night 2's derivative-free recipe (fast16, truncated draws, hub chaining, local x4,
embedding x4, WSD). 5090 for track A and the context tests, 4070 for MoE.

## Summary

1. **Leak-free local keys/values/gates/value embeddings works** (`--local-separate`): 13-49x less variance per
   draw, -0.06 nat at L4 (2 seeds), -0.076 nat at L8.
2. **Real Dust does not learn to copy context; noise-free Dust does.** Draw noise, not bias, blocks context use.
   Leak-free scoring with gamma 1 is the first Dust run that learns the copy (second-half loss 2.65 -> 0.98).
3. **MoE does not help Dust at 8 sequences per step**, though it helps backprop (with AdamW). The gap shrinks
   with more tokens per step (+0.098 at 8 sequences, +0.032 at 32, +0.007 at 64) but never flips.

## Track A (5090): leak-free local scoring

Dust's local estimator perturbs every token's key at once, so each key's score picks up all other keys' noise.
A softmax with one shifted logit has a closed form, so every token's key (or value) can be scored on its own
perturbation, all T experiments in one T x T pass per draw (formula in the plan below). Exact finite difference,
no derivatives, zero leakage inside the local estimator.

**Correctness and noise** (check_separate.py, bench_separate.py; bp-trained L4, exact own-token o@ targets):
matches brute-force one-key reruns to 1e-15. Variance cut at 8 / 32 / 128 draws:

| site | cut |
|---|---|
| block 0 keys | 35 / 33 / 30x |
| block 1 keys | 49 / 47 / 44x |
| block 0 values | 27 / 26 / 24x |
| block 1 values | 32 / 32 / 28x |
| gate | 16 / 13 / 15x |
| value embeddings | 32 / 32 / 29x |

These match the predicted 1 + kappa.

**Training, L4-D256** (best recipe; seeds 42 / 7):

| arm | val | wall time |
|---|---|---|
| baseline (local x4) | 2.733 / 2.749 (mean 2.741) | 1043 s |
| leak-free, local x4 | 2.679 / 2.683 (mean **2.681**) | 1677 s* / 1248 s |
| leak-free, local x1 | 2.683 / 2.692 (mean **2.688**) | 973 s (seed 7; 1250 s for seed 42 on a shared GPU) |
| control: old estimator, local x1 | 2.907 | 1113 s |
| leak-free, local x1, gamma 1 | 2.714 | 977 s |

*the 1677 s run shared the 5090 with short tests. At equal local draws the leak-free estimator wins by 0.22 nat;
with 4x fewer local draws it still beats the baseline by 0.05 and is faster. The gain appears late (identical at
step 1000), during the WSD decay. On TinyStories gamma 1 is slightly worse than .98.

**Training, L8-D512** (night 2 reference 2.918, 3552 s): leak-free local x4 **2.842** (4439 s, -0.076);
local x1 2.934 (3430 s). At L8 the 4x local draws still matter.

**Why it works only late, and what is left:** night 4's oracle runs put the largest noise cost in self-attention
internals (0.26 nat), but at local x4 their own estimator noise was already small; the rest comes from their
targets (attention-output errors chained from the residual-writer estimates). Leak-free scoring removes the local
part; the writer-site noise remains the root.

## Context use (5090): can Dust learn to copy?

TinyStories models barely use context, so they hide both leakage and bias. Two testbeds:

- **MQAR** (Zoology's associative recall, answers scored only): the repo's SGD stays exactly at chance under
  backprop (128 / 32 / 8 pairs; 2048 or 64 values; lr .2 to 1.5). AdamW learns it (1.71 vs chance 4.16).
  Not usable with this optimizer.
- **TinyStories-repeat** (256 story tokens, then the same 256 again): backprop with SGD learns to copy.

| run | val | first half | second half (copyable) |
|---|---|---|---|
| backprop | 1.104 | 2.126 | 0.049 |
| noise-free Dust, gamma 1 | 1.169 | 2.216 | 0.090 |
| noise-free Dust, gamma .98 | 1.342 | 2.401 | 0.251 |
| real Dust, best recipe | 2.752 | 2.823 | 2.645 |
| real Dust, stock | 3.055 | 3.046 | 3.014 |
| leak-free, local x1, gamma .98 | 2.649 | 2.756 | 2.490 |
| leak-free, local x1, gamma 1 | 2.035 | 3.045 | **0.983** |

- Noise-free Dust's bias costs only 0.065 nat at gamma 1, but 0.24 at gamma .98: the copy source is 256 tokens
  back, beyond gamma .98's ~50-token horizon.
- Real Dust learns none of the copy. Leak-free scoring makes a long horizon affordable, and with gamma 1 Dust
  learns most of it; plain next-token modeling (first half) gets worse, 2.82 -> 3.05.
- Measured on this bp-trained model: writer kappa stays small (block 0 .05-.07); key/value kappa 9-13.

## Track B (4070): MoE

Fine-grained routed MoE (`yoco/moe.py`, `--moe E,k,h,shared`): sigmoid scores, top-k, dropless, shared expert,
auxiliary-loss-free bias balancing. Dust on MoE perturbs the routed experts' hidden units and the selected
router logits, with routing held fixed during those draws.

- **Backprop, 3000 steps, matched active hidden (1024):** with SGD every MoE ties or trails dense (more, smaller
  experts are worse). With AdamW every MoE beats dense by 0.03-0.08 (e.g. lr 3e-3: dense 1.773, E8k2 1.711).
- **Dust, 1500 steps** (E8 k2 h384 + shared 256): MoE 3.085 vs dense 2.987. A dense MLP as wide as the MoE's total
  (3328) gets 3.126. Under AdamW: MoE 3.219, dense 3.044.
- **Why:** a weight's SNR goes as (tokens using it) / A. At matched active params A is the same as dense and each
  expert weight sees k/E = 1/4 of the tokens. Against an equally wide dense layer, A and tokens per weight shrink
  together and cancel, which is why MoE ~ wide dense.
- **Tokens per step** (same total tokens; does MoE catch up out of the noise-limited regime?):

| sequences per step | steps | dense | MoE | MoE - dense |
|---|---|---|---|---|
| 8 | 1500 | 2.987 | 3.085 | +0.098 |
| 32 | 375 | 3.159 | 3.191 | +0.032 |
| 64 | 188 | 3.297 | 3.304 | +0.007 |

  The gap closes steadily but MoE never pulls ahead; at a fixed lr, fewer and larger steps also make both
  worse.
- **Fixed 1500 steps, more tokens** (tinystories-long, 72k rows, same validation): 32 seqs/step dense 2.643,
  MoE 2.728 (+0.085, vs +0.098 at 8). More tokens per step barely closes the gap, so the shrinkage above was
  mostly both models undertrained at few steps. 48 seqs/step and 8-seq references on the same data: running.
  The 64-sequence runs first ran out of memory on the 4070 (the logits at 64 x 2 draws). They now use gradient
  accumulation over microbatches of 32 (`--microbatch-size`), statistically the same estimator. The lr stays at
  0.25 at every batch, so absolute losses rise with fewer, larger steps; the gap is what matters.

**Note:** on 2026-11-05 at about 16:55 the 4070's power limit went from 100 W to 200 W, partway through
the 1500-step x 32-sequence dense run. 4070 wall times after that are not comparable with earlier ones.

## Next (queued: yoco/queue_n6.sh, 5090, starts after the 4070 queue)

1. Leak-free scoring for the YOCO cache (`--cache-separate`): copying goes through the cache, whose estimator
   still has cross-token leakage (cache-key kappa 10-40). Bench first, then the copy task.
2. Which sites need the long gamma horizon: gamma 1 everywhere taught the copy but hurt plain next-token
   modeling (first half 2.82 -> 3.05). gamma 1 on the cache only vs the self-decoder only.
3. Confirm the copy result: second seed, then L8 with backprop and noise-free references.
4. (Code, not queued) fp16 / fused separate scoring, to remove its ~25% time cost at L8.

## Files

- `yoco/dustx.py`: `separate_scores`, `--local-separate` path, MoE sites, `oracle_re`, microbatch accumulation.
- `yoco/moe.py`: MoE layer and Dust hooks. `yoco/train.py`: `--moe`, `--opt`, `--local-separate`,
  `--microbatch-size`, no-loss target encoding (token + 8192).
- Benches: `check_separate.py`, `bench_separate.py`, `halves.py`. Data: `make_mqar.py`, `make_repeat.py`.
- Queues: `queue_n4.sh` (oracles), `queue_n5*.sh` (5090), `queue_moe*.sh` (4070).

# Plan as written at the start of the night (branch `kappa-moe`)

The user allowed both GPUs tonight (the 4070 is power-capped at 100 W). The 5090 (index 0) runs the kappa work and
the 4070 (index 1) runs the MoE work. Timings are never compared across GPUs.

## Why

- Night 4's oracle runs (FINDINGS.md) show draw noise in the self-decoder's attention internals (q/k/v/gate/ve)
  costs the most loss of any site group: 0.26 nat of the 0.68 nat noise gap at L4. Those sites also have the
  largest leakage kappa (20-45 here, 10-1000+ in Qwen3; see ~/Dust Leakage and Bias in Qwen3.md).
- Dust's SNR per draw is about 1/(A (1+kappa)), where A is the number of jointly perturbed dimensions with
  nonzero gradient. Two levers follow: cut kappa (track A) and cut A (track B, MoE).

## Track A (5090): leak-free local keys and values

Dust's local estimator perturbs every token's key at once, so each key's score picks up all other keys' noise.
Perturbing one key changes one logit per query, and a softmax with one shifted logit has a closed form:

    o'_u = (o_u + P_us (e^d - 1) v_s) / (1 + P_us (e^d - 1)),   d = q_u . a_s / sqrt(dh)
    score_us = e_u . (o'_u - o_u) = P_us (e^d - 1) / (1 + P_us (e^d - 1)) * (e_u . v_s - e_u . o_u)

All T one-key experiments cost one T x T pass per draw, the same as a current local draw. Values: o'_u - o_u =
P_us a_s exactly. This is a finite difference of the real attention function (no derivative), with zero
cross-token leakage inside the local estimator. Predicted variance cut: 1 + kappa_local.

Steps: implement (`--local-separate`), check against brute-force one-key perturbation, bench per-site cos to the
exact local target at fixed draws (expect the 1+kappa gap), then L4 training runs (2 seeds) at the current local
draw count and at 1/4 of it.

## Track B (4070): MoE for Dust

With routing, A at an expert's hidden site is k*h (active units), independent of E, so parameters can grow
without growing per-token noise. First establish that MoE helps under backprop at this scale (small-scale MoE
gains are usually modest), against a dense model at matched active parameters, same data and steps:

- dense: hidden 1024 (=4d)
- E8 k2 h512; E16 k2 h256 + shared 512; E32 k4 h128 + shared 512; E64 k4 h128 + shared 512
- SGD as in the rest of this repo; experts see k/E of the tokens, so also try an expert lr multiplier.

Design follows DeepSeekMoE / V4.1 scaled down: fine-grained experts, a shared expert, sigmoid scores with top-k
normalisation, dropless routing, auxiliary-loss-free bias balancing (derivative-free already). Pointers used:
fine-grained MoE scaling laws (Ludziejewski/Krajewski et al., ICML 2024: granularity > 1 helps at every budget),
OLMoE (arXiv 2409.02060: 64 experts, top-8, dropless; shared expert not useful there), nanoGPT MoE study
(arXiv 2402.13089), DeepSeek-V4.1-Flash report (CED = YOCO variant; MoE 384/6 + 1 shared everywhere).

Then Dust on MoE: per-expert hidden sites (perturb only routed experts) and router-score sites, compared against
Dust on the dense model and against the backprop MoE results above.

Results go in FINDINGS.md as they land.
