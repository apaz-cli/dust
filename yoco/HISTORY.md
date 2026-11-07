# Training YOCO without backprop: what we tried and what we learned

This is the story of about a week of experiments on training YOCO language models with Dust, a zeroth-order
(derivative-free) training method. It is written to be read on its own. Numbers come from `FINDINGS.md`, which
logs every run; the code is in this repository (`yoco/`).

## The question

Dust trains a network without backpropagation. For each linear layer it adds small random noise to that layer's
output, separately for every token, reruns the forward pass, and scores each token's noise against how that
token's own loss changed. Averaged over enough draws, this recovers a gradient. The method is attractive
precisely because it needs no derivatives anywhere, which is why one rule held throughout: anything that used a
derivative was allowed only as a measuring instrument, never as part of a proposed method.

YOCO ("you only cache once") splits a transformer into a self-decoder, which builds one shared key-value cache,
and a cross-decoder, whose layers all read that cache. The appeal for Dust is that the cross-decoder is clean:
given the cache, each token's computation there is independent of every other token's, so Dust's per-token
estimate is exact for those layers. The trouble is concentrated in the self-decoder and the cache.

An agent swarm had produced a pile of ideas about combining the two. The job was to test them, find better
zeroth-order methods for YOCO, and understand where Dust's errors come from.

Everything ran on one RTX 5090 (a second GPU, an RTX 4070, joined late for one line of work). The main testbed
was small: a 4-layer, 256-wide YOCO model (6.2M parameters) on TinyStories, 512-token sequences, 8 sequences per
step, 3000 steps, about 12M tokens in all. On that setup backprop reaches a validation loss of 1.926, and stock
Dust reaches 3.131. Most of what follows is about closing that gap.

## Night 1: taking the error apart

Before improving Dust, we needed to know what its error is made of. Dust's estimate can be wrong in four
separable ways:

- Bias: Dust scores token t's noise only against token t's own loss. But token t's activations also affect
  every later token, through attention. Dust misses that credit, no matter how many draws it averages.
- Leakage (κ): in one draw every token is perturbed at once, and earlier tokens' noise leaks into token t's
  loss through attention. This adds noise to t's score.
- Curvature (η): the loss is not linear over the perturbation's size.
- Rounding: loss differences computed in bf16 lose precision.

The most useful tool built that night was a noise-free version of Dust. It computes exactly what Dust would
estimate on average, using a two-route backward pass as a ruler. Training with it shows what Dust's bias alone
costs: about 0.1 nat (2.060 against backprop's 1.926). Real Dust sat a full nat higher, so draw noise made up
almost all of the gap.

Other measurements from that night shaped everything after:

- bf16 rounding was a hidden noise source. At deep sites, rounding the reward inflated the estimate's variance
  19 to 100 times. Computing rewards in fp32 removes it.
- Leakage is small on the residual stream and large inside attention. κ was about 0.1 for the layers that
  write to the residual stream in block 0, near zero deeper, and 3 to 45 for keys and values.
- The bias lives in the first two layers, jointly. Giving only one early layer its full credit barely helped;
  giving both did. Cross-token circuits need a two-layer credit chain, and in YOCO those two layers are the
  self-decoder.

Where draws go turned out to matter more than how many there are. Attention internals (queries, keys, values,
gates) are cheap to probe locally, by rerunning one attention operation instead of the whole model. Spending 4x
more draws there, plus 4x on the attention-output errors ("hubs") and on the embedding, took the loss from 3.131
to 2.777.

Several ideas failed and were dropped. Giving credit in proportion to current attention made the model unable
to learn to attend anywhere new. Giving early layers a discounted share of later tokens' losses cost more in
noise than it bought back in bias. A "handoff" scheme that trained the self-decoder against a shared estimate
of the cache's error diverged.

One observation from that night turned out to matter much later: noise-free Dust learned to use context almost
as well as backprop did, while real Dust's use of context flattened out after about 32 tokens. The guess at the
time was that the credit through keys and values was being lost to noise.

## Night 2: making it fast

The best recipe from night 1 was better but 1.6 times slower than stock Dust. Night 2 went after speed without
changing what was estimated:

- fp16 matrix multiplies with an fp32 residual stream and fp32 rewards ("fast16"), matching TF32's quality;
- a fused, compiled loss head, using the cross-entropy fix from Barron et al.;
- **truncated draws**: a perturbation at block l cannot change blocks before l, so each draw reruns only from
  block l, starting from the clean pass's recorded state (bit-exact);
- batched head draws.

A draw went from 2.12 ms to 0.55 ms on average. The next change was hub chaining: instead of full forward passes to
estimate attention-output errors, estimate them with cheap local probes of the output projection, scored against
the residual-writer estimates. With a warmup-stable-decay learning-rate schedule on top, the best derivative-free
recipe reached 2.741 (two seeds) in about 1040 seconds, where night 1's best needed 2128 seconds to reach 2.777.

The rest of the night tested ideas that sounded promising and mostly were not:

- Optimal draw allocation. The theory works (spend in proportion to the square root of noise over cost), and
  we measured both quantities. Running with the measured allocation scored 2.762 against 2.760: no gain. Dust
  runs far below its critical batch size, so where draws go matters less than how noisy each one is.
- Draw schedules. Spending more draws late in training was worse (2.790): early noise slows learning.
- NVFP4 weights. Keeping an fp32 master copy and quantizing for the forward pass cost nothing (2.767).
  Without the master copy, stochastic rounding alone lost 0.19 nat. More interesting: quantizing only the
  perturbation in each matrix multiply ("delta forward") keeps 99% of the estimate's quality at NVFP4, and real
  NVFP4 kernels run 5x faster than fp16 on the 5090. This was never put into a training run.
- Pooling estimates across depths cut the per-step error by up to 16 times on a benchmark and still lost in
  training, because it introduced a bias, and momentum averages away noise but never bias.
- Crediting earlier draws through later keys and values fixed the bias in principle (it worked with exact
  key/value errors) and failed in practice, because Dust's key/value errors were too noisy to use.

The same day, night 1's code was used to test two further noise reductions: perturbing only the MLP hidden units
that are active ("live"), and a control variate that predicts each site's error from the head's error (computed
by finite differences on the cached logits) and subtracts the predictable part. The prediction only shapes the
control variate, so the estimate's expectation does not change. Together they took the night-1 recipe from 2.765
to 2.464 (two seeds). They were then left out of every faster recipe until the final runs, which is where they
turned out to be the largest gain of all.

## Night 3: does the bias grow with scale?

The article that started this asked whether Dust's bias stays small as models grow. We trained 4-, 8- and
12-layer models (6M, 32M and 95M parameters) on the same 12M tokens, with backprop, noise-free Dust and real
Dust at each size. The bias cost stayed flat at about 0.1 nat; the noise cost rose from 0.68 to 0.96. We also
confirmed the article's estimate of the extra leakage from discounted credit: γ²/(1−γ²), about 23 at γ = 0.98,
matched to within 2% on residual-stream sites.

The conclusion that night was that bias is not the problem and noise is. That conclusion was wrong in an
instructive way.

## The turn: models that use context

Measurements on pretrained Qwen3 models (0.6B to 32B, at 4096 tokens of context) told a different story. In models
that actually use long context, leakage at the early residual-stream layers is 3 to 8 (against 0.1 here), and the
own-token target that Dust aims at explains under 1% of the per-token gradient energy in the first 70% of the layers at 32B. TinyStories models
barely use context, so they hide both problems: they have almost no cross-token gradient to get wrong. And κ
falling during Dust training, which had looked like good news, actually means the model is learning to ignore
the context Dust cannot credit.

Two things followed. A diagnostic run gave one group of sites its noise-free update at a time. The
self-decoder's attention internals came out as the single largest noise cost: making them noise-free alone took
the loss from 2.733 to 2.473. And we needed a testbed where using context is the whole point.

## Night 5: leak-free keys and values

Dust's local estimator for keys perturbs every token's key at once, recomputes attention, and scores the change
at each output position. Output u reads every key up to u, so each key's score is contaminated by every other
key's noise. That contamination is κ, and for keys it is 20 to 45 here.

The fix rests on one observation. Perturbing a single key changes a single logit in each query's row, and a
softmax with one shifted logit has a closed form:

    o'_u = (o_u + P_us (e^δ − 1) v_s) / (1 + P_us (e^δ − 1)),   δ = q_u · Δk_s / √d

So the effect of perturbing each key *alone* on every output can be computed for all keys at once, in one
attention-sized pass. Values are simpler still: changing value s changes output u by P_us times the change. This
is an exact finite difference of the real attention function, with no derivative involved. Each key still gets
credit from every query that reads it; only the cross-talk between keys is gone.

It checked out against brute-force recomputation to 1e-15, and cut the variance per draw 13 to 49 times,
matching 1 + κ. In training:

| | 4 layers (2 seeds) | 8 layers |
|---|---|---|
| previous best recipe | 2.741 | 2.918 |
| leak-free, same local draws | 2.681 | 2.842 |
| leak-free, 4x fewer local draws | 2.688 (7% faster) | 2.934 |

At equal local draws the leak-free estimator wins by 0.22 nat. The gain appears late in training, during the
learning-rate decay, when the update is small and noise matters most.

## Can Dust learn to use context?

The first context testbed, multi-query associative recall, could not be used: with the plain SGD this repository
uses, even backprop stays exactly at chance on it (AdamW learns it). The second worked. Each training row is 256
tokens of a story followed by the same 256 tokens again. The second half can be predicted by copying, which
needs attention to look back 256 tokens.

| run | loss on the first half | loss on the copied half |
|---|---|---|
| backprop | 2.13 | 0.05 |
| noise-free Dust, γ = 1 | 2.22 | 0.09 |
| noise-free Dust, γ = 0.98 | 2.40 | 0.25 |
| real Dust, best recipe | 2.82 | 2.65 |
| leak-free self-decoder, γ = 1 | 3.05 | 0.98 |
| + leak-free cache, γ = 1 | 3.03 | 0.43 (second seed 0.50) |
| + leak-free cache, γ = 1 in the self-decoder, 0.98 on the cache | 2.93 | **0.16** |

This is the central result. Real Dust does not learn to copy at all; noise-free Dust learns it nearly as well
as backprop. So for using context, draw noise is what stands in the way, which is what night 1 had guessed.
Removing leakage from the key and value estimators, and extending the credit horizon (γ = 1; the copy source is
farther back than γ = 0.98's roughly 50-token horizon), lets Dust learn most of the copy. The long horizon
matters in the self-decoder's attention; on the cache it barely helps. One cost remains: plain next-token prediction (the
first half) gets worse with γ = 1, 2.93 against 2.82.

## The MoE detour

The other line of work asked whether mixture-of-experts layers could help Dust by shrinking the number of
dimensions perturbed per token: with routing, each token only touches k of the E experts. The MoE layers follow
DeepSeek's design, scaled down: fine-grained experts, a shared expert, top-k routing, and load balancing by a
bias nudged according to expert load, which needs no gradient.

Under backprop, MoE beat a dense model of the same active size by 0.03 to 0.08, but only with AdamW; with SGD it
lost. Under Dust, MoE lost to dense with either optimizer, by 0.03 to 0.09, at 8, 32 and 48 sequences per step.

The reason is a cancellation. A weight's estimate sums error times input over the tokens that use it, so its
signal-to-noise ratio goes roughly as the number of tokens using it divided by the number of jointly perturbed
dimensions. At matched active size, routing leaves the perturbed dimensions unchanged and gives each expert
weight a quarter of the tokens. Against a dense layer as wide as the MoE's total, routing shrinks both by the
same factor. As predicted, Dust on the MoE scored about the same as Dust on a dense layer of equal total width.
For Dust, what MoE offers is cheaper draws; the noise per weight stays where it was.

## The recipe

The best derivative-free recipe for this setting, as command-line flags to `yoco/train.py`:

    --arch yoco --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only
    --emb-scale 4 --wsd 0.2 --local-separate --cache-separate --local-scale 1 --live-fc --cv

(`--local-scale 4` at 8 layers.) On a task that needs long-range context, add `--gamma 1 --cache-gamma 0.98`.

## Final runs

The last runs confirmed the leak-free results and brought back night 1's two forgotten noise reductions, the
live-unit hidden jitter and the head-error control variate. That combination is the largest single gain of the
whole project:

| | 4 layers (2 seeds) | 8 layers |
|---|---|---|
| stock Dust | 3.131 | 3.163 |
| night 2 best recipe | 2.741 | 2.918 |
| + leak-free scoring (self-decoder and cache) | 2.687 | 2.887 |
| + live hidden units and control variate | **2.364** | **2.352** |
| backprop | 1.926 | 1.860 |

The control variate also works without leak-free scoring (2.404 at 4 layers), so leak-free scoring adds about
0.04 on top on TinyStories. On the copy task the combination is what makes the difference:

| copy task, 4 layers | first half | copied half |
|---|---|---|
| backprop | 2.13 | 0.05 |
| noise-free Dust, γ = 1 | 2.22 | 0.09 |
| real Dust, night 2 best recipe | 2.82 | 2.65 |
| leak-free, γ = 1 in the self-decoder | 2.93 | 0.16 |
| + live hidden units and control variate | **2.60** | **0.08** |

Real Dust now copies as well as noise-free Dust, and its plain next-token loss on the first half is better than
any earlier Dust run. Without the control variate the γ = 1 configuration diverged on a second seed. At 8
layers, without the control variate, a lower learning rate (0.15) took the copied-half loss from 1.46 to 0.55;
the 8-layer copy run with the control variate has not been done.

## Where it stands

Open problems:

- Residual-writer noise. The layers that write to the residual stream are now the main noise source, and
  their noise also feeds the attention targets. Leakage there is small in these models but 3 to 8 in models
  that use context.
- The first-half cost of a long credit horizon: γ = 1 lets Dust copy and makes ordinary next-token
  prediction worse.
- Scale: the copy result at 8 layers needs a lower learning rate, and has not been run with the control
  variate.
- Speed: leak-free scoring runs in fp32 and costs about 25% more time at 8 layers; NVFP4 delta forwards are
  measured but unused.

Some lessons about the method of working, for whoever continues:

- Small models on simple data can hide the problem you are studying. TinyStories made bias look harmless.
- Per-step benchmarks mislead. Momentum averages noise across steps but never bias, so a change that cuts
  per-step error can still lose in training (pooling did).
- A noise-free version of the method is the most useful ruler there is: it separates bias from noise exactly.
- Reread the old results. The best idea of the week was found on day two and then dropped by accident.
