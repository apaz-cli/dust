# Findings log (L4-D256, TinyStories T=512, 3000 steps x 8 seqs, population-256 recipe, RTX 5090)

## Exactness checks (yoco/check_exact.py, yoco/check_expected.py)
- YOCO cross-decoder: dL_u/dY_t = 0 exactly for u != t (c_fc, both cross blocks).
- Exact head error, local attention VJPs (self + cross/cache), handoff = residual probe + cache pullback: all match autograd to ~1e-7.
- Expected-DUST two-route backward: gamma=1/full == backprop (7e-6); own-route writer grads == brute-force own-token target (1e-6).

## Backprop references (val loss @3000)
- L4: gpt lr0.2 1.920, yoco lr0.2 1.923 (lr0.1: 2.017 / 2.021). L8: gpt 1.866 (lr0.1), yoco 1.808 (lr0.2).

## Bias of the expected DUST target (yoco/measure_bias.py), weight-space cos to true gradient
- Backprop-trained GPT: block0 0.91-0.94, block1 0.98-0.99, block2 0.99-1.0, block3 1.0; wte 0.81.
- Backprop-trained YOCO: block0 0.90-0.94, block1 0.98-0.996 (own) -> 1.0000 with handoff target; cross blocks 1.0; wte 0.75.
- DUST-trained GPT (q1/gpt-dust): every block site >= 0.98, blocks 1-3 future share 0.0000 -> bias hides itself (model barely mixes tokens).
- Per-token future share understates weight-space error: GPT bp block1 c_proj fut 0.005 but ||G-Gown||/||G|| = 0.21.
- Old swarm claim "wte wcos worsens with scale 0.84/0.60/0.58": means over steps>=500 are 0.69/0.67/0.71 (sd 0.13). No trend.

## Context use (yoco/context_use.py): val loss by position, gain = loss[1:8] - loss[256:512]
- backprop gpt lr0.2: gain 0.77, still improving to 512.
- DUST (Oct L4, q1 gpt-dust): gain 0.39, flat after ~32 tokens.

## Noise-free expected DUST (method ed), lr 0.1
- gpt gamma 0: 2.529 | 0.98: 2.118 | 0.995: 2.109 | 0.999: 2.105 | 1: 2.107   (bp lr0.1: 2.017)
- yoco gamma 0.98: 2.136 | 1: 2.120 | 1+handoff: 2.112 | 0: 2.502
- Real DUST gpt: 3.115.  => bias costs ~0.1 nat, draw noise ~1.0 nat.
- Context: ed gamma0.98 gain 0.686 (bp 0.701), ed gamma0 gain 0.398 flat after 32 == real DUST's profile.
  Hypothesis: in real DUST the k/v credit is lost to noise in the local ZO draws.
- lr 0.2 (bp: gpt 1.920, yoco 1.923): ed gpt g0 2.505 | g0.98 2.036 | g0.995 2.025 | g0.999 2.021 | g1 2.022
  ed yoco g0.98 2.045 | g1 2.021 | g1+handoff 1.997 | g0 2.495
  => bias cost vs bp: gpt stock 0.116, yoco stock 0.122, yoco vjp 0.098, yoco handoff 0.074.

## Real DUST, L4, seed 42, lr 0.25 (q1)
- gpt-dust 3.115 (context gain 0.389) | yoco-dust 3.128 (gain 0.364)
- yoco-vjp 2.863 (gain 0.524, loss still falling at 256-512): exact local attention VJP instead of ZO local draws: -0.265 nat.

## L8 noise-free (seed 42): bp gpt 1.866/1.938 (lr .1/.2), bp yoco 1.866/1.808
- ed gpt g.98 1.963/1.896 | g1 1.951/1.875 | yoco g.98 1.976/2.030 | yoco g1 1.961/? | yoco handoff 1.952/1.968
- yoco-handoff (real DUST, 32 handoff draws): DIVERGED at step 128.
  200-step diag (val@200): vjp no-handoff 3.52 | handoff h128 3.77 | h512 3.59 | h32 lr0.15 4.60.
  => shared noisy g_X makes the handoff recipe worse than plain vjp even with 512 extra draws.
- gpt-vjp 2.843 (gain 0.511) vs gpt-dust 3.115: -0.272. Same as yoco (-0.265). Not YOCO-specific.
  yoco-vjp 2.863 vs gpt-vjp 2.843: YOCO no better than GPT under DUST.

## Cost (yoco/bench_local.py, in-process medians per step, shared GPU)
- L4: direct draws 0.95 s | ZO local (q,k,v,gate,ve) 63 ms | exact local VJP 11 ms | ZO head 203 ms | exact head 4 ms
- L8: direct 3.23 s | ZO local 217 ms | VJP 29 ms | ZO head 214 ms | exact head 5 ms
- "head = 77% of budget": 77% of draw COUNT; ~17% of step time at L4, ~6% at L8.

## Seeds (42, 7, 11), yoco/summarize_ed.py -> runs/ed_summary.json
- L4 best lr 0.2: bp gpt 1.919+-.015, bp yoco 1.926+-.019 | ed gpt g.98 2.048+-.012, g1 2.026+-.004 |
  ed yoco g.98 2.060+-.021, g1 2.031+-.020, handoff 2.028+-.028
- L8 lr 0.1 (lr 0.2 unstable at L8: bp sd .04, ed-yoco-g1 s42 diverged at step 10): bp gpt 1.863+-.003, bp yoco 1.860+-.013 |
  ed gpt g.98 1.968+-.005, g1 1.952+-.002 | ed yoco g.98 1.971+-.005, g1 1.954+-.006, handoff 1.950+-.002
- => noise-free bias cost 0.09-0.13 nat at both sizes; gamma .98->1 buys ~0.02; YOCO / handoff within seed noise.
- gpt-exacthead (real DUST): 3.116 vs 3.115. No quality change; saves ~17% step time at L4.
- seed 7: yoco-vjp 2.847, gpt-vjp 2.864 -> yoco-vjp 2.855+-.011, gpt-vjp 2.853+-.015 (n=2).

## Constraint from the user (07:15): no derivatives anywhere. The local VJP, exact head, cache pullback and
## handoff pullback are derivative methods -> diagnostics only, not proposals. Derivative-free follow-up:
## more ZO local draws (cheap), fewer ZO head draws. Oracle arms are diagnostics.
- gpt-vjp-head (diagnostic) 2.780.

## Derivative-free local estimator bench (yoco/bench_local_zo.py), bp-trained YOCO, exact own-token o@ errors
- Stock budget is mostly noise: cache k (12 draws) cos ~0.07, cache v (20) cos ~0.2; dense estimator unbiased (rel ~ 1/sqrt(N)).
- Designs at 256 draws: sparse position masks (P=4/16/64), antithetic, Rademacher: no gain (sparse worse).
  => noise is cross-row interference, and isolating rows costs exactly the samples it saves.
- Credit kernel vs full (gamma=1) local gradient: gamma 1.0 worst (noise), 0.9 best of gammas at small N.
  Attention-matched credit w_ut = min(1, c*T*P_ut), c=0.01 (forward probabilities only, derivative-free):
  N=16 cos: cache k .148 (g.98: .072), self0 k .291 (.106), self0 v .366 (.150), self1 v .478 (.194)
  N=256 cos: cache v .603-.746 (g.98 .590), self0 k .716 (.386), self0 v .786 (.513), self1 v .838 (.641)
- c in 0.001..0.01 about equal in cos; smaller c shrinks magnitude.
- Full pipeline (ZO o@ errors, hub draws 6): o@ estimate cos to exact only .02-.10 per step (96 draws: .09-.38).
  Every local design then has per-step cos ~0 (momentum averages across steps); per-step bench can't rank them.
- YOCO direct cache scoring (jitter cache rows, reward with attention-weighted later losses, no o@ stage):
  rel 13-52 (attn) / 44-455 (gamma) vs 5-7 via hub route -> rejected; each token loss carries every read row's jitter.

## bf16 rounding noise in DUST rewards (yoco/bench_hub_precision.py, q1/yoco-dust, 96 o@ draws, cos to exact o@ error)
- bf16 stock sigma: o@0 .379 o@1 .198 o@2 .154 o@3 .091 | sigma x2: .390 .291 .238 .151 | x4: .391 .357 .323 .235
- fp32 (and TF32 matmuls): .395 .397 .395 .397 at every sigma -> the draw-noise floor.
- => bf16 rounding of loss differences inflates deep-hub variance up to ~19x (cos^2 .008 vs .158). TF32 costs ~1.4x per draw.
- Writers / MLP (yoco/bench_site_precision.py, 64 draws, sigma 0.2, cos to exact own-token error):
  YOCO bf16: b0 .262/.171/.230  b1 .104/.145/.105  b2 .048/.095/.047  b3 .030/.082/.028 (attn.c_proj/mlp.c_fc/mlp.c_proj)
  YOCO fp32: every block .30-.33 / .173 / .30-.33  (the d/Z floor; c_fc is 4x wider)
  GPT  bf16: b2 .063/.122/.065  b3 .036/.089/.035 ; fp32 .31-.33 / .17 / .31-.33
  => bf16 rounding inflates deep-writer variance 50-100x. General DUST finding, not YOCO-specific.
     The repo's draw allocation (38,32,16,10 for L4) puts few draws where bf16 estimates are near-useless.

## Training, derivative-free arms (L4, seed 42)
- gpt-local4 (4x ZO local draws): 3.011 vs gpt-dust 3.115 (-0.10), context gain 0.386.
- yoco-attn (attention-matched credit, c=.01): 3.170 vs yoco-dust 3.128 (+0.04 worse), gain 0.217.
  Per-step alignment improved on the bench, but credit proportional to current attention cannot teach a query
  to attend somewhere new (rich-get-richer bias). Dropped.
- yoco-fp32 (TF32 forwards): 3.054 vs 3.128 (-0.074), gain 0.361, +18% wall time.

## Error budget (yoco/error_budget.py; weight space, / ||G_true||^2; noise & rounding per draw; N=16, R=4)
- DUST-trained YOCO: bias^2 <= .006 (wte .056); noise/draw 30-190; rounding/draw 31 (b0) .. 1330-5458 (b2-b3 writers)
  -> in the regime DUST actually trains in, bf16 rounding dominates deep sites ~100x.
- backprop-trained YOCO: bias^2 b0 .17-.30, b1 .01-.05, cross 0, wte .50; noise/draw 55-345; rounding 2-15.
- Per step noise >> bias everywhere; bias persists across steps, noise averages.
- yoco-local4 3.034 (-0.094, gain .410) | yoco-fp32-local4 2.977 (-0.151, gain .442, 1745 s vs 1360 s stock).

## Orthogonal probe directions (yoco/bench_orthogonal.py; attn.c_proj, d=256, central differences, cos to own-token error)
- sigma 0.02 + TF32 is rounding-limited; use sigma 0.2. fp32 ('highest'): Gaussian N=64 .447, N=256 .706 (= sqrt(N/(N+d)));
  orthogonal N=64 .50 (= sqrt(N/d)), N=256 (full basis) 1.000 at every site of DUST-trained YOCO.
- bp-trained (context-using): full basis gives YOCO .958/.998/1.000/1.000, GPT .963/.998/.999/1.000 by block
  -> leakage only blocks exactness at block 0 (1-cos^2 ~ 8%). TF32 costs a little at depth (.935 at YOCO block 3).
- Below a full basis the gain is cos^2 x (1 + N/d): 4-15% at stock budgets; decisive only when draws per site >= width.

## Exact leakage kappa (measure_bias.py, bp-trained, 2 seqs): sum_{s<t}||dL_t/dY_s||^2 / ||dL_t/dY_t||^2
- residual-stream sites (writers, c_fc, c_q): block0 ~.09-.11 both archs; block1 .001-.005; deeper GPT <=.001, YOCO cross 0 exactly.
- k/v weights: 3-45 (GPT b0 k 19.9 v 21.2, b1 k 30.6, b2 k 22.9, b3 k 8.4); YOCO self b0 k 23.3 v 24.7, b1 k 44.7.
- YOCO cache k 11.8, v 2.9 (like GPT's deep k/v). wte .82 (YOCO) / .76 (GPT): worst direct site (also bias^2 .5, 10 draws).
- yoco-local16-hd4 3.003 (-0.125, 1897 s). yoco-fp32-local4-hub4 2.842 (-0.286, gain .456, 2023 s = 1.49x stock):
  derivative-free, matches the VJP diagnostic (2.855).
- gpt-fp32 3.059 vs gpt-dust 3.115 (-0.056). yoco-dust seed 7: 3.135 -> stock YOCO 3.131 +- .005 (n=2).

## Where the bias lives (noise-free, own-token streams only in chosen layers, gamma 1, lr .2, seeds 42/7; cost vs bp)
- GPT single: L0 +.011, L1 +.028, L2 -.007, L3 0. YOCO single: L0 noisy (-.044/+.028), L1 +.037.
- GPT pairs: [0,1] +.09 | [2,3] +.009 | [0,1,2] +.11 | [1,2,3] +.041 | all ~+.10.
- => bias is superadditive and lives in the first two layers jointly (cross-token circuits need a 2-layer credit chain).
   In YOCO those are the self-decoder, which YOCO does not change.
- Residual credit (res_gamma on blocks 0-1 + wte), per-step error vs TRUE gradient (fp32, stock draws): monotone worse
  with res_gamma in both bp- and DUST-trained YOCO (block0 c_proj err 2.20 -> 2.84 (.5) -> 5.6 (.8) on bp-trained).
  Per-step metric undervalues bias (momentum averages ~20 steps); training arms res .5/.8 queued.
