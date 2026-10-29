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
