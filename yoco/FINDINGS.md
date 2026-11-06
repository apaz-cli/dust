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

## YOCO split depth (L4, lr .2, seeds 42/7): backprop / noise-free (gamma 1, own streams incl. cache) / bias cost
- n_self 1: 1.950 / 2.053 / .103 | n_self 2: 1.916 / 2.020 / .104 | n_self 3: 1.892 / 2.019 / .127
- deeper self-decoder helps backprop; noise-free DUST is flat 2->3, so bias grows with self-decoder depth.
  n_self=1 does not remove bias (the cache stream is own-token too: still a 2-step chain). Default split is fine.
- (8x hub: yoco-fp32-local4-hub8 2.832 vs hub4 2.842; diminishing.)
- Note: DustX.zo_self_local assumes even n_self (has_ve parity); fix before real-DUST runs with odd n_self.
- Residual credit in training: yoco-fp32-local4-hub4-res5 2.907 vs 2.842 without (+0.065, gain .406). Variance cost wins;
  early-layer bias (~.09 noise-free) is not recoverable by a gamma discount at stock draws.
- Compute control: yoco-direct2 (2x writer+hidden draws) 3.068 at 2067 s vs fp32-local4-hub4 2.842 at 2023 s -> -0.226 at equal wall time.
- yoco-fp32-local4-hub4-emb4 (4x embedding draws): 2.777 (-0.065 vs 2.842), 2128 s.
- Seed 7 of fp32-local4-hub4: 2.885 -> best recipe 2.863 +- .03 (n=2) vs stock 3.131 +- .005.
- L8-D512 stock YOCO DUST: 3.163 (3846 s, context gain .331); bp L8 yoco 1.808-1.866.
- L8-D512 best (fp32+local4+hub4+emb4): 2.904 vs stock 3.163 (-0.259), 7305 s vs 3846 s (1.9x), context gain .358.
- Queue 5: emb16 2.744 (vs emb4 2.777, 2644 s) | fp32-local4-hub4-flat 2.952 (vs 2.842: stock front-loaded allocation is better)
  | yoco-hub4 bf16 alone 3.016 (vs stock 3.131).

# Night 2 (2026-10-30)
## Perf (L4 YOCO): fixed CE (2610.31642) + compiled head loss + flash causal + fast16 (fp16 matmuls/attention, fp32 stream,
## fp32-output lm_head) + truncated draws (bit-exact) + batched head draws.
- draw cost: 2.12 ms (TF32, morning) -> 0.97 fast16 -> 0.55 truncated (avg over sites). fast16 quality = TF32 (deep c_proj cos .297 vs .298).
- step: stock bf16 0.37 s; old best (TF32, untruncated) 0.605 s; best fast16+trunc 0.455 s -> after head batching ~0.39 s.
- nrep (draws per forward) 2..16: no speedup (GPU saturated at 8192 tokens).
- NVFP4 _scaled_mm works on 5090: 82 us vs fp16 421 us (5x), fp8 1.9x (16384x512x2048).
## KV-credit augmentation (reward += <e_strict K/V, delta K/V> for upstream draws), bp-trained YOCO, 24 reps:
- ZO K/V errors (hub4/local4): step err block0 c_proj 2.21 -> 10.85, c_fc 7.9 -> 42, wte 2.4 -> 218 (fails: K/V error noise).
- exact K/V errors (autograd ruler): cos to TRUE 0.752 -> 0.837 (b0 c_proj), .560 -> .620 (c_fc), wte .558 -> .778, step err ~same.
  => correct bias fix; bottleneck is K/V error quality (per-step cos ~.01-.05).
## Allocation probe (alloc_probe.py), best-fast step 500: B_g ~ 300-3000 per group (N << B everywhere: linear regime).
## Linear values (gray zone: uses attention's linearity in V, like reassemble uses W's): value-error cos to target
- ZO local (4x): cache .035, b0 .043, b1 .044 | linear: cache .157, b0 .207, b1 .223 (per step, hub4). No V draws.
- KV credit with linear values, values only: b0 c_proj cos-to-true .752 -> .786, step err 2.21 -> 2.35; wte err +75% (excluded).
## Delta forward (quantize the perturbation, not the activations), cos to exact own-token error, 64 draws:
- Dust-trained YOCO: fast16 .332/.332/.323/.297 | fp8 deltas .332/.333/.332/.332 | nvfp4 deltas .328/.330/.329/.329
- bp-trained YOCO:   fast16 .320/.333/.333/.333 | fp8 .320/.333/.333/.333 | nvfp4 .315/.328/.329/.329
  => NVFP4 deltas keep ~99% of estimate quality; fp8 deltas even remove fast16's residual deep rounding.
## Allocation drift (stationary model, x stock-best draws): attn.0 .74 -> 1.03 -> 1.21 (steps 500/1000/2000), attn.1 .14 -> .17 -> .60,
   head 1.31 -> 1.22 -> 1.02, wte 1.65 -> 1.51 -> 1.24; predicted gain of reallocation 25% -> 22% -> 15%.
## Residual pooling (writers + embedding pooled across depths; data-driven scales)
- per-step error vs own target (Dust-trained YOCO): 2.5-16x lower for writers, 8.8x for wte (oracle bound similar).
- TRAINING: best-pool 2.827 (EMA 2.791) vs best-fast 2.760 -> worse. Joint damage of pooled noise is LOWER (0.0010 vs 0.0027),
  so not correlated-noise amplification; it is BIAS: pooled error floors at 0.6-0.9 |G|^2 (per-token scale ratios between depths
  vary; one global scale per pair misweights tokens). Momentum averages noise, never bias.
## Local chaining (c_fc via local MLP probes, o@ via local c_proj probes, 256 cheap draws, scored vs pooled targets):
- per-step error vs stock direct draws: c_fc 2.2-8.6x lower, o@ 9-25x lower; no hidden or hub full forwards.
- unpooled variant (scored vs own writer estimates) is unbiased; queued (chain-own).
- WSD 0.2 on best-fast: 2.735 vs 2.760 (EMA under WSD: 2.747, no gain).
- chain-a (pooled local chaining, 256 local draws): 2.80-2.82 (EMA 2.756) at step 2500-2600, then DIVERGED (NaN, step 2640);
  0.59 s/step (local MLP probes at width 1024 cost more than the draws they replace). Dropped.
- Unpooled chaining (bench, Dust-trained): c_fc 2-4x WORSE than direct draws (writer noise spreads through the MLP map);
  o@ 4x better at block 0, equal deeper. -> only hub chaining kept as a candidate (chain-hubs).
## Training results (night 2, YOCO L4-D256, single seed)
- chain-hubs (hub chaining vs own writer targets, local4, emb4, fast16): 2.771 @ 1040 s; + WSD 0.2: 2.733 @ 1043 s (best).
- stock-fast (stock recipe, fast16 + fixed CE + truncation): 3.054 @ 828 s (EMA 3.023) vs night-1 stock 3.131 @ 1360 s.
- alloc-stat (stationary-model allocation): 2.762 @ 1339 s (EMA 2.743) vs 2.760 @ 1150 s: no gain; cost model underestimated time.
- best-ramp-wsd2 (equal total draws, spent late): 2.790 vs best-fast-wsd2 2.735: worse; early noise slows learning.
- quant-master (NVFP4 weights in forward, fp32 master): 2.767 vs 2.760.
- best-linv (linear values, gray zone): 2.699 @ 1084 s (EMA 2.661) vs best-fast 2.760 -> best of night 2 at constant lr.
- quant-sr (NVFP4 weights stored, SR updates, no master): 2.951 (EMA 2.917): stable but -0.19 vs master copy (2.767).
- stock-wsd2 (bf16 stock + WSD): 3.082 @ 779 s vs 3.131 night-1 stock.
- best-b16-lr35 (16 seqs x 1500 steps, lr x sqrt2; equal data and compute): 2.911 vs 2.760 -> more, smaller steps win.
- chain-hubs-wsd2 seed 7: 2.749 -> best derivative-free recipe 2.741 +- .008 (n=2) @ ~1041 s.
- linv-hubs-wsd2 (gray zone + hub chaining + WSD): 2.706 @ 973 s (linear values alone, constant lr: 2.699, EMA 2.661).

## Night 2 addendum: L8-D512 scale check (3000 steps, fast16, 5090 only)

| Arm | val (EMA) | seconds | night 1 reference |
|---|---|---|---|
| best ZO recipe | 2.918 (2.924) | 3552 | best 2.904 @ 7305 s |
| stock Dust | 3.077 (3.134) | 2778 | stock 3.163 |

The best ZO recipe keeps a 0.16 nat lead over stock at L8 and matches night 1's best in under half the time.
The stock run gains about 0.09 from fast16 and the CE fix alone.

## Queue 6 (seed 42, idle 5090): noise PR #5 live-unit c_fc jitter (--live-fc) and PR #6 control variate (--cv)
- --cv: per-site ridge memory from the head error (central finite differences on cached logits, mapped by lm_head)
  to the site's estimated error; draws measure error - prediction. Estimate's expectation is unchanged for any memory.
- Live c_fc share on a DUST-trained YOCO (q5/yoco-hub4): .11 / .27 / .39 / .45 by block.
- best = fp32+local4+hub4+emb4 (2.777, q3):  +live 2.683 | +cv 2.546 | +live+cv 2.457 (1442 s)
  | +live+cv with half the draws at blocks 0-1 and wte (writers/hidden 20,16,16,10, emb 20) 2.565 (1242 s).
- stock + live + cv 2.847 (936 s) vs stock 3.131. Noise-free Dust 2.060, backprop 1.926.
- Reward variance removed by the memory at step 3000 (best+live+cv): writers b0 .32, b1 .71, b2 .84, b3 .91;
  c_fc .17-.26; hubs .33/.64/.81/.92; wte .33. Lowest at the early sites, where the draws cost most.
  Stock bf16: deep writers only .05-.25 (rounding noise is unpredictable).
- Wall times are on an idle GPU; earlier queues shared it. Timing reruns of stock and best are in queue 7.

## Queue 7: seed 7 and idle-GPU timing
- best+live+cv: s42 2.457, s7 2.470 -> 2.464 +- .009. half early draws: 2.565, 2.556 -> 2.560 +- .006.
- Reruns on the idle GPU: best (fp32+local4+hub4+emb4) 2.765 in 1424 s (q3: 2.777); stock 3.130 in 832 s.
- Wall time vs best: +live+cv 1442 / 1535 s (+1-8%, run-to-run ~6%); half 1242 / 1214 s (0.86x).
  => half-draw arm is 14% cheaper than best and 0.21 lower. stock+live+cv 2.847 in 936 s (1.12x stock).
- Gap to noise-free Dust (2.060): best .71 -> live+cv .40.

## Queue 8 (seed 42, idle 5090): early-draw floor with live-fc + cv
- fp32+local4+hub4 + live+cv, blocks 0-1 writers/hidden and wte draws cut:
  quarter (10,8 / emb 10) 2.751 in 1039 s | eighth (6,4 / emb 6) 2.804 in 935 s.
  => quarter matches the old best (2.765, 1424 s) at 0.73x the time.
- stock + live + cv, half early draws (20,16 / emb 6): 2.934 in 708 s vs stock 3.130 in 832 s (0.85x time, -0.20).

## Night 3: scaling ladder (L4-D256 6.2M / L8-D512 32M / L12-D768 95M params; 3000 steps x 8 x 511 tokens = 12M tokens)
Fixed token budget at every size, so the larger models are undertrained (L12 bp is no better than L8 bp).
- L12 lr: Dust lr .25 diverges by step 9; .15 stable (40-step check, better than .1). bp/ed: .05 beats .1.

### Noise-free bias cost (ed vs bp, val @3000; L4/L8 from night 1, 3 seeds; L12 single seed, lr .05)
| size | bp | ed g.98 (stock target) | ed g1 | bias cost g.98 / g1 |
|---|---|---|---|---|
| L4  | 1.926 | 2.060 | 2.031 | .134 / .105 |
| L8  | 1.860 | 1.971 | 1.954 | .111 / .094 |
| L12 | 1.871 | 1.981 | 1.961 | .110 / .090 |
=> the bias cost does not grow from 6M to 95M params; if anything it shrinks slightly.
- L12 lr .1: bp 1.908, ed g1 1.968, ed g.98 2.123 (g.98 less stable at the higher lr).

### Gamma-credit leakage split (measure_bias.py --gamma; bp-trained YOCO, 2 seqs; k_b = s<t, k_a = s>t)
- Article estimate g^2/(1-g^2) truncated at T=511: 23.06 (g.98). Residual-stream sites measure k_a 21-23.5 at L4/L8/L12.
- k/v and cache: k_a lower (5-22; late tokens have little future, so a_ss shrinks), k_b high (6-43).
- Own-token target cos to true grad (bp-trained): k/v .17-.35, cache k .17 v .31, wte .75-.78, writers .93-.98.
- g.98 target cos: block k/v .83-.95, value_embeds .88-.94, cache .75-.83, wte .97-.98.
- L4 gamma sweep (cos / k_b / k_a): cache_k g.9 .53/17.5/2.5 | g.98 .77/32.8/9.6 | g.995 .95/52.9/29.8 | g.999 .997/65/62
  block0 c_k g.9 .79/7.5/4.1 | g.98 .94/10.2/22.0 | g.995 .99/13.2/77 | g.999 1.0/14.8/180.
  => the cache's k_a grows ~3x slower in g than residual sites': cache can take a higher gamma for less noise.

### Real Dust at L12 (lr .15, fast16, single seed; another process shared the 5090 from ~04:00, step time stayed flat)
| size | bp | ed g.98 | best ZO | stock | best wall s | stock wall s |
|---|---|---|---|---|---|---|
| L4  | 1.926 | 2.060 | 2.741 (2 seeds) | 3.054 | 1041 | 828 |
| L8  | 1.860 | 1.971 | 2.918 | 3.077 | 3552 | 2778 |
| L12 | 1.871 | 1.981 | 2.942 | 3.108 | 8982 | 7160 |
- Noise cost (best ZO - ed g.98): .68 / .95 / .96. Bias cost .13 / .11 / .11. Noise share of the gap .84 / .90 / .90.
- Best-vs-stock lead: .31 / .16 / .17 nat. Best costs 1.25x stock's wall time at every size.
- L12 best-zo step: 3.0 s (L8 1.18 s, L4 0.35 s).
- Dust-trained L12 (g.98 target): cache k cos .77 (same as bp-trained), block k/v .86-.91, wte .996.

### Per-site gamma in training (L4 best recipe, cache k/v gamma .995, others .98)
- seeds 42/7: 2.729 / 2.756 (mean 2.742) vs one gamma 2.733 / 2.749 (2.741). No change: the target cos gain
  (.77 -> .95) is lost in the cache estimator's draw noise.
- L12 bp-trained own-token cos: block k .17-.18, v .24, cache k .16 v .30, wte .76 (same as L4/L8).

## Night 4: where draw noise costs loss (L4 best ZO recipe 2.733; one group given its noise-free expected update)
- self-decoder q/k/v/gate/ve 2.473 (-.26) | self writers 2.567 (-.17) | wte 2.607 (-.13) | self c_fc 2.612 (-.12)
  | cross blocks 2.684 (-.05) | head 2.705 (-.03) | cache k/v 2.725 (-.01). Oracle also removes the o@-target noise.
- Idea saved for later: leak-free local k/v. Perturb each key alone; softmax with one shifted logit has a closed form,
  o'_u = (o_u + P_us(e^d - 1) v_s) / (1 + P_us(e^d - 1)), d = q_u.a_s/sqrt(dh); values: do_u = P_us a_s.
  All T one-key experiments cost one attention-sized T x T pass per draw, no extra model passes. Removes local kappa:
  ~1+kappa = 20-50x less variance per draw for L4 keys/values (10-1000x on Qwen3 keys).

## Night 5 (plan and reasons: yoco/NIGHT5.md)
### Track A: leak-free local k/v/gate/ve (--local-separate), check and bench
- check_separate.py: closed form == brute-force one-key / one-value attention rerun to 1e-15 (float64).
- bench_separate.py on bp-trained L4 YOCO, exact own-token o@ targets, weight-space rel. error vs reference:
  variance cut (stock / separate) at N = 8 / 32 / 128 draws: block0 k 35/33/30x, block1 k 49/47/44x,
  block0 v 27/26/24x, block1 v 32/32/28x, gate 16/13/15x, value_embeds 32/32/29x. Matches 1 + kappa.
- Cost: about +20% step time at the recipe's local draw count (fp32 T x T einsums; not optimized).

### Track B: MoE under backprop at L4-D256 (RTX 4070; val / EMA, seeds 42 / 7; active hidden matched to dense 1024)
| config | SGD lr .2 (repo optimizer) | note |
|---|---|---|
| dense | 1.901 / 1.937 (EMA 1.836 / 1.881) | |
| E8 k2 h512 | 1.922 / 1.894 (EMA 1.878 / 1.850) | ties dense within seed noise (~.02-.04) |
| E16 k2 h256 + shared 512 | 1.958 / 1.978 | expert lr x4: 1.939 / 1.956 |
| E32 k4 h128 + shared 512 | 1.965 / 1.962 | |
| E64 k4 h128 + shared 512 | 2.003 / 1.976 | expert lr x4: 1.973 / 1.949 |
- Under SGD, more and smaller experts are worse, and an expert lr multiplier only partly helps. AdamW runs
  (queue_moe2.sh) check whether this is the optimizer: each expert sees k/E of the tokens, so SGD steps shrink.
- Vectorised MoE forward (padded bmm) matches a naive loop to 1e-6 and is 2-6x faster than the first version.
- Dust on MoE implemented: routed pre-activations (site mlp.moe, width k*h) and selected router logits (mlp.gate,
  width k) join each layer's hidden draws, with the clean routing held fixed during those draws; expert weight,
  router and shared-expert estimates are reassembled from the site errors. Needs a shared expert (its c_proj
  output is the layer-output site). 4070 speed: ~3.0 s/step vs 1.7 s dense (while sharing the GPU).

### Context testbeds (5090, backprop, SGD lr .2)
- MQAR (data/mqar*, only answers scored): SGD stays exactly at chance (P=128, 32, 8; 2048 or 64 values; lr .2 to
  1.5). AdamW learns P=8/64-vocab to 1.71 (chance 4.16). The repo's SGD cannot learn associative recall here.
- TinyStories-repeat (data/tsrepeat: 256 story tokens then the same 256): SGD backprop learns to copy. Val 1.104,
  first half 2.19, second half 0.045. Writer kappa stays small (block 0 .05-.07, own cos .94); k/v kappa 9-13.
  Queued: does Dust learn the copy? (queue_n5b.sh: noise-free g.98 / g1, best ZO, stock).

### Track A training (first read)
- sep-l4-s42 at step 1000: 3.168 vs 3.169 for the same recipe without the flag. The local estimator's own noise was
  not the binding part: at local-scale 4 it already matched the exact local derivative (night 1). The self-attention
  oracle gain (.26) comes mostly from the o@ targets, which are chained from the writer estimates.
- AdamW (bp, betas .9/.95, no wd), val (EMA), seed 42:
  lr 1e-3: dense 1.837 (1.763) | E8k2 1.760 (1.688) | E16k2+s 1.782 (1.718) | E32k4+s 1.772 (1.703)
  lr 3e-3: dense 1.773 (1.664) | E8k2 1.711 (1.600) | E16k2+s 1.745 (1.638) | E32k4+s 1.713 (1.611)
  => under AdamW every MoE beats dense by .03-.08 nat at matched active parameters (the expected small-scale
  MoE gain). Under SGD it does not. SGD is the reason, and Dust uses SGD: queue_moe4 tries Dust with AdamW.
- Final (L4 best recipe, seed 42; baseline without the flag 2.733, seed 7 2.749):
  sep-l4 (local draws x4) 2.679 (EMA 2.689) | sep-l1 (x1, 4x fewer local draws) 2.683 (EMA 2.690).
  Same at step 1000; the gap opens late (WSD decay), when the update is small and noise matters most.
  Wall time on a shared GPU: 1677 s (l4) / 1250 s (l1) vs 1043 s; the separate path's fp32 T x T einsums are not
  optimized (draw-independent e.v / e.o terms now cached).
- Seeds 42 / 7 (val): sep-l4 2.679 / 2.683 (mean 2.681) | sep-l1 2.683 / 2.692 (mean 2.688, 973 s for seed 7 with the
  cached terms, vs 1043 s baseline) | baseline l4 2.733 / 2.749 (2.741) | control stock-l1 2.907 (seed 42).
  => at equal local draws the leak-free estimator wins by .22 nat; at 4x fewer local draws it still beats the
  recipe by .05 nat and is faster. New best derivative-free L4 recipe: best + --local-separate --local-scale 1.

### Does Dust learn to copy? (TinyStories-repeat, L4 YOCO, 3000 steps; val first half / second half)
| run | val | first half | second half |
|---|---|---|---|
| backprop (SGD .2) | 1.104 | 2.126 | 0.049 |
| noise-free Dust, gamma 1 | 1.169 | 2.216 | 0.090 |
| noise-free Dust, gamma .98 | 1.342 | 2.401 | 0.251 |
| real Dust, best ZO recipe | 2.752 | 2.823 | 2.645 |
| real Dust, stock | 3.055 | 3.046 | 3.014 |
- Noise-free Dust learns the copy (bias costs .065 nat at gamma 1, .24 at gamma .98: the copy source is 256
  tokens back, beyond gamma .98's ~50-token horizon). Real Dust does not learn it at all: draw noise, not bias,
  is what keeps a Dust-trained model from using context here. Matches the user's point that falling kappa in
  Dust training signals a model that stops using context.
- Queued (queue_n5d): the same task with leak-free local scoring at gamma .98 and 1.

### Dust on MoE (4070, 1500 steps, E8 k2 h384 + shared 256 vs dense hidden 1024; val)
| | dense | MoE |
|---|---|---|
| backprop SGD .2 | 2.172 | 2.259 |
| backprop AdamW 3e-3 | 1.986 | 1.966 |
| Dust best recipe, SGD | 2.987 | 3.085 |
| Dust best recipe, AdamW 1e-3 | 3.044 | 3.219 |
- MoE loses under Dust with either optimizer, by more than backprop's gain. Likely reason: a weight's estimate sums
  error x input over the tokens that use it, so its SNR goes as (tokens using it) / A. Routing shrinks A per token
  by about the same factor as the tokens per expert (k/E), so per-weight SNR does not improve, while at matched
  active params each expert weight sees 4x fewer tokens than a dense weight. Routing sparsity cuts the cost of a
  draw, not its noise per weight. Test queued (queue_moe5): dense MLP as wide as the MoE's total hidden (3328).
- Wide dense (hidden 3328 = the MoE's total), 1500 steps: backprop SGD 2.190 (dense 2.172, MoE 2.259); Dust 3.126
  (dense 2.987, MoE 3.085). As predicted, Dust-MoE ~ Dust-wide: at a fixed draw budget, more MLP parameters hurt
  Dust whether dense or routed. MoE at equal total size is no worse than dense-wide and cheaper per draw, so its
  value for Dust is draws per FLOP, not lower noise per weight.

### Track A at L8-D512 (night 2 reference: best ZO 2.918, 3552 s)
- sep-l4 2.842 (EMA 2.855, 4439 s): -0.076 nat. sep-l1 2.934 (3430 s): at L8 the 4x local draws still matter.

### Copy task with leak-free local scoring (local-scale 1; first half / second half)
- rep-sep-g98 2.649: 2.756 / 2.490 | rep-sep-g1 2.035: 3.045 / **0.983** (old recipe 2.823 / 2.645).
- With separate scoring and gamma 1, Dust learns to copy for the first time (second half 2.65 -> 0.98). The first
  half gets worse (2.82 -> 3.05): a gamma-1 horizon helps the copy and costs plain next-token modeling.
- TinyStories, sep-l1 gamma 1: 2.714 vs 2.683 at gamma .98 (TinyStories needs no long horizon).

### Dust MoE vs dense by tokens per step (4070, same total tokens, lr .25)
- 8 seqs x 1500: dense 2.987, MoE 3.085 (+.098) | 32 x 375: dense 3.159, MoE 3.191 (+.032) | 64 x 188: dense 3.297, MoE 3.304 (+.007)
  (first attempt out of memory; now gradient accumulation over microbatches of 32, --microbatch-size).
- The gap closes with tokens per step but does not flip at equal total tokens.
- 16:55 2026-11-05: user raised the 4070 power limit 100 W -> 200 W during dust1500-b32-dense (step ~288). Wall times on the 4070 from here on are not comparable to earlier 4070 runs; losses are unaffected.
- queue_moe7 v1 crashed at step 750 (24k training rows ran out at 32 seqs/step); relaunched on tinystories-long (72k rows, same validation) with 32 and 48 seqs/step, plus 8-seq references on the same data.
- Fixed 1500 steps, tinystories-long: 32 seqs/step dense 2.643, MoE 2.728 (+.085) vs 8 seqs/step +.098 (24k data).
  4x the tokens per step barely moves the gap. The shrinking gap at equal total tokens (+.098/+.032/+.007) was
  mostly both models being undertrained at 375/188 steps, not MoE escaping the noise limit.
- 48 seqs/step x 1500: dense 2.586, MoE 2.659 (+.073). Gap by tokens per step at 1500 steps: 8 -> +.098,
  32 -> +.085, 48 -> +.073. Shrinking slowly (~.004 per 8 seqs; zero near ~150-200 seqs/step if linear), no flip.
