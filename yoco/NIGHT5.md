# Night 5 plan: two tracks (branch `kappa-moe`)

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
