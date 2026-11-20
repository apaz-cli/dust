"""Train the GPT or YOCO model on TinyStories (T=512) with DUST variants or backprop.

Mirrors ~/git/dust/leakage/train_small.py: the population-256 10M recipe truncated to n_layer blocks,
3000 steps of batch 8, validation every 100 steps on the 128-row split.
"""
import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import sys
import time

# Under mlsweep, runs on different GPUs compile at the same time; a shared inductor/triton cache then races
# ("Unknown key: 'cubin'"). Give each run its own cache before torch is imported.
if 'MLSWEEP_RUN_NAME' in os.environ:
    _cache = f"/tmp/torchinductor_{os.environ['MLSWEEP_RUN_NAME']}"
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', _cache)
    os.environ.setdefault('TRITON_CACHE_DIR', _cache + '/triton')

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import GPTConfig, make_optimizer, recipe  # noqa: E402
from yoco import expected  # noqa: E402
from yoco.dustx import DustX  # noqa: E402
from yoco.model_yoco import build  # noqa: E402
from mlsweep.logger import MLSweepLogger  # noqa: E402  (no-op outside an mlsweep worker)


class TensorAdam(torch.optim.Optimizer):
    """Adam with one second moment per tensor instead of per coordinate. It still adapts the step to each tensor's
    gradient size (as MoE experts that see few tokens need), but within a tensor keeps the update's direction, so
    coordinates where Dust's noise was removed (the control variate) are not rescaled back up."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g['betas']
            for p_ in g['params']:
                if p_.grad is None:
                    continue
                s = self.state[p_]
                if not s:
                    s['t'], s['m'], s['v'] = 0, torch.zeros_like(p_), 0.0
                s['t'] += 1
                s['m'].mul_(b1).add_(p_.grad, alpha=1 - b1)
                s['v'] = b2 * s['v'] + (1 - b2) * p_.grad.float().square().mean().item()
                mh = s['m'] / (1 - b1 ** s['t'])
                vh = s['v'] / (1 - b2 ** s['t'])
                p_.add_(mh, alpha=-g['lr'] / (vh ** 0.5 + g['eps']))


def split(batch, noloss=8192):
    """Inputs and targets from token rows; tokens >= noloss are real tokens (minus noloss) with no loss."""
    x = (batch[:, :-1] % noloss).contiguous()
    y = batch[:, 1:].contiguous()
    return x, torch.where(y >= noloss, torch.full_like(y, -1), y)


@torch.no_grad()
def evaluate(model, rows, batch_size=32):
    total = 0.0
    for batch in rows.split(batch_size):
        x, y = split(batch)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            total += model(x, y).item() * len(batch)
    return total / len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=['gpt', 'yoco'], default='gpt')
    p.add_argument('--method', choices=['dust', 'bp', 'ed'], default='dust')
    p.add_argument('--head', choices=['zo', 'exact'], default='zo')
    p.add_argument('--local', choices=['zo', 'vjp'], default='zo')
    p.add_argument('--handoff', action='store_true')
    p.add_argument('--handoff-draws', type=int)
    p.add_argument('--gamma', type=float)
    p.add_argument('--population', type=int, default=256, choices=[256, 1024, 4096])
    p.add_argument('--res-gamma', type=float, default=0.0, help='discounted later-loss credit for early-block sites')
    p.add_argument('--res-blocks', type=int, default=2)
    p.add_argument('--emb-scale', type=int, default=1, help='multiply the token-embedding draws')
    p.add_argument('--direct-scale', type=int, default=1, help='multiply writer and hidden draws')
    p.add_argument('--flat-alloc', action='store_true', help='same writer/hidden draws in every block (same total)')
    p.add_argument('--fp32', action='store_true', help='same as --precision tf32')
    p.add_argument('--precision', choices=['bf16', 'tf32', 'fast16'], default='bf16',
                   help='DUST forwards: bf16 (stock), tf32 (fp32 activations) or fast16 (fp16 matmuls, fp32 stream and logits)')
    p.add_argument('--alloc', help='explicit draws: JSON {"writers": [...], "hidden": [...], "hubs": [...], '
                                   '"embedding": n, "head": n}; applied after every scale flag')
    p.add_argument('--quant', choices=['none', 'master', 'sr'], default='none',
                   help='NVFP4 block weights: master = fp32 master + quantized forwards; sr = stored in NVFP4, SR updates')
    p.add_argument('--linear-v', action='store_true', help='value errors from the linear attention map (no V draws)')
    p.add_argument('--chain', type=int, default=0, help='local chaining with this many cheap local draws (implies --pool)')
    p.add_argument('--chain-own', action='store_true', help='local chaining against each site\'s own (unpooled) estimate')
    p.add_argument('--chain-hubs-only', action='store_true', help='chain only the hub (o@) errors')
    p.add_argument('--pool', action='store_true', help='residual pooling of writer and embedding estimates')
    p.add_argument('--kv-credit', action='store_true', help='cross-token credit through estimated K/V errors')
    p.add_argument('--draw-ramp', action='store_true',
                   help='same total draws, spent late: 0.75x for the first 60%% of steps, then linear 0.75x -> 2x')
    p.add_argument('--ema', type=float, default=0.995, help='track an EMA of the weights (eval only); 0 disables')
    p.add_argument('--clip', type=float, default=2.0,
                   help='adaptive clipping: scale a step whose gradient norm exceeds CLIP x its running average (0: off)')
    p.add_argument('--microbatch-size', type=int, default=0,
                   help='Dust: gradient accumulation over microbatches of this many sequences, one optimizer step')
    p.add_argument('--batch-seqs', type=int, default=8, help='sequences per step')
    p.add_argument('--wsd', type=float, default=0.0, help='WSD: decay the lr linearly to 0 over this final fraction')
    p.add_argument('--no-truncate', action='store_true', help='rerun the whole model for every draw')
    p.add_argument('--cache-gamma', type=float, help='credit decay for the YOCO cache k/v (default: --gamma)')
    p.add_argument('--credit', choices=['gamma', 'attn'], default='gamma')
    p.add_argument('--attn-c', type=float, default=0.01)
    p.add_argument('--local-scale', type=int, default=1, help='multiply the ZO local (q, k, v, gate, ve) draws')
    p.add_argument('--head-div', type=int, default=1, help='divide the ZO head draws')
    p.add_argument('--live-fc', action='store_true', help='MLP hidden draws jitter only live units (noise PR #5)')
    p.add_argument('--cv', action='store_true', help='control variate from a head-error-keyed credit memory (PR #6)')
    p.add_argument('--cv-beta', type=float, default=0.9)
    p.add_argument('--cv-own', action='store_true', help='control variate also keyed on the clean stream entering the site\'s block')
    p.add_argument('--cv-neighbor', action='store_true', help='control variate also keyed on a nearby site\'s error from this step (unbiased)')
    p.add_argument('--cv-warmup', type=int, default=20)
    p.add_argument('--hub-scale', type=int, default=1, help='multiply the o@ (attention output) draw counts')
    p.add_argument('--ed-own-layers', type=int, nargs='*', help='ed: only these layers keep own-token streams')
    p.add_argument('--sim-noise', type=float, default=0.0, help='ed only: noise simulator, Dust-sized noise C d / N per site')
    p.add_argument('--sim-kv-head', action='store_true', help='noise simulator: size key/value noise by the head width')
    p.add_argument('--sim-tail', help='noise simulator: heavy tails at the same variance, "DF,elem" or "DF,token"')
    p.add_argument('--sim-floor', type=float, default=0.0, help='noise simulator: noise floor, a share of the mean per-token error')
    p.add_argument('--sim-cv', help='noise simulator: JSON of control-variate shares per site (cv_explained of a recipe run)')
    p.add_argument('--init', help='checkpoint folder to start from (its step3000.pt)')
    p.add_argument('--halves-every', action='store_true', help='copy task: log both halves\' loss at every evaluation')
    p.add_argument('--n-self', type=int, help='YOCO self-decoder depth (default n_layer // 2)')
    p.add_argument('--freeze-attn', action='store_true', help='residual-stream draws hold later attention and the cache clean')
    p.add_argument('--readout', action='store_true', help='cross-token readout: each draw also scores the closed-form change of later tokens\' attention outputs from its own keys and values (needs --chain --chain-own --chain-hubs-only)')
    p.add_argument('--readout-scale', type=float, default=1.0, help='weight of the readout term in each draw\'s reward')
    p.add_argument('--readout-mode', choices=['credit', 'leak', 'both'], default='credit',
                   help='readout: credit later tokens\' loss changes to t (credit), cancel the predicted leakage into each '
                        'token\'s own reward (leak), or both (symmetric readout)')
    p.add_argument('--anchor', type=int, default=0,
                   help='anchored draws: sites below this self-decoder block rerun only up to it, scored against the '
                        'error there (needs --chain without pooling)')
    p.add_argument('--anchor-scale', type=int, default=1, help='multiply the anchored sites\' draws')
    p.add_argument('--anchor-boost', type=int, default=1, help='multiply the draws of the writers that give the anchor error')
    p.add_argument('--anchor-warmup', type=int, default=0, help='steps of ordinary draws before anchoring')
    p.add_argument('--readout-warmup', type=int, default=0, help='steps of plain Dust before --readout turns on')
    p.add_argument('--cache-separate', action='store_true', help='leak-free scoring for the YOCO cache k/v too')
    p.add_argument('--local-separate', action='store_true', help='leak-free local k/v/gate/ve scoring (closed form)')
    p.add_argument('--oracle-re', help='regex on parameter names given their noise-free expected update')
    p.add_argument('--oracle', nargs='*', default=[], help='groups given their noise-free expected update')
    p.add_argument('--moe', help='routed MoE MLPs: "E,k,h,hs" (experts, top-k, expert hidden, shared hidden)')
    p.add_argument('--opt', choices=['sgd', 'adamw', 'adam_tensor'], default='sgd',
                   help='adamw: bp reference optimizer (lr --adam-lr); adam_tensor: Adam with one second moment per tensor')
    p.add_argument('--adam-lr', type=float, default=3e-3)
    p.add_argument('--adam-beta2', type=float, default=0.95)
    p.add_argument('--adam-eps', type=float, default=1e-8)
    p.add_argument('--router-lr-mult', type=float, default=1.0, help='lr multiplier for MoE router weights (AdamW)')
    p.add_argument('--moe-balance-rate', type=float, default=1e-3, help='step of the load-balancing bias per update')
    p.add_argument('--moe-lr-mult', type=float, default=1.0, help='lr multiplier for routed expert weights')
    p.add_argument('--moe-fix', action='store_true', help='Dust MoE: live units for the routed experts; no control variate at expert/router sites')
    p.add_argument('--router-cap', type=float, default=0.0, help='MoE: soft cap on router logits, c*tanh(x/c) (0: off)')
    p.add_argument('--router-wd', type=float, default=0.0, help='MoE with AdamW: decoupled weight decay on the router only')
    p.add_argument('--orth', nargs='*', default=[], help='orthogonal draws at sites ending with these (e.g. mlp.moe mlp.c_fc)')
    p.add_argument('--expert-probes', type=int, default=0, help='Dust MoE: expert hidden errors from N local probes against the layer output error (0: full draws)')
    p.add_argument('--moe-cv', action='store_true', help='with --moe-fix: control variate fitted per expert at the expert sites')
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True, help='Dust: torch.compile the reruns and head draws')
    p.add_argument('--compile-force', action='store_true', help='keep the compiled forward even if the self-check fails')
    p.add_argument('--moe-grouped', action=argparse.BooleanOptionalAction, default=True,
                   help='MoE: one grouped matmul over the slots sorted by expert (fixed shapes: compiles); '
                        '--no-moe-grouped for the padded per-expert batches (host sync, runs eagerly)')
    p.add_argument('--moe-dense', action='store_true', help='MoE: compute every expert for every token, keep the routed ones (fixed shapes: compiles)')
    p.add_argument('--compile-backend', default='inductor', help='torch.compile backend for the model forward (diagnostics)')
    p.add_argument('--nan-debug', action='store_true', help='print NaN counts of eager and compiled forwards on one batch, then exit')
    p.add_argument('--moe-layers', type=int, nargs='*', help='layers that get the MoE (default all)')
    p.add_argument('--n-layer', type=int, default=4)
    p.add_argument('--n-embd', type=int, default=256)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--lr', type=float, default=0.25)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--save-at', type=int, nargs='*', default=[])
    p.add_argument('--nvfp4', action='store_true',
                   help='Dust: reruns through the linear layers as a delta forward on NVFP4 tensor cores (yoco/nvfp4.py; '
                        'Blackwell GPUs only)')
    p.add_argument('--nvfp4-experts', action='store_true', help='with --nvfp4: the routed experts too (delta forward)')
    p.add_argument('--nvfp4-skip', type=int, nargs='*', default=[], help='with --nvfp4: blocks kept in full precision')
    p.add_argument('--count-flops', action='store_true',
                   help='count the floating-point operations of step 2 (run uncompiled) and of one forward pass; '
                        'logged as flops_per_token, flops_forward_per_token, flops_ratio; breakdown in flops.json')
    p.add_argument('--noise-probe-every', type=int, default=0,
                   help='every N steps (and every step of --diag-window), measure the noise and bias of Dust\'s '
                        'estimate per site and per weight against exact references (yoco/noise_probe.py; 0: off)')
    p.add_argument('--diag-window', type=int, nargs=2, default=None, metavar=('FIRST', 'LAST'),
                   help='also log the per-part and routing diagnostics at every step from FIRST to LAST')
    p.add_argument('--save-every', type=int, default=500, help='also save the weights every N steps (0: off), for replaying a run')
    p.add_argument('--data-dir', type=Path, default=Path('data/tinystories'))
    p.add_argument('--output', type=Path, help='run folder (default: $MLSWEEP_RUN_DIR under mlsweep)')
    args = p.parse_args()

    device = torch.device('cuda')
    torch.set_float32_matmul_precision('high')
    heads = args.n_embd // 64
    train = torch.load(args.data_dir / 'train.pt', weights_only=True).long()
    val = torch.load(args.data_dir / 'validation.pt', weights_only=True).long().to(device)
    # the sequence length (and with it the short attention window, half of it) follows the data's rows
    config = GPTConfig(sequence_len=train.shape[1] - 1, n_layer=args.n_layer, n_head=heads, n_kv_head=heads,
                       n_embd=args.n_embd)
    settings = recipe('10M', args.population)
    n = config.n_layer
    fit = lambda xs: (xs + [xs[-1]] * n)[:n]  # deeper than the recipe's 8 layers: repeat the last (deepest) count
    settings = replace(settings, writers=fit(settings.writers), hidden=fit(settings.hidden), hubs=fit(settings.hubs),
                       lr=args.lr, **({'gamma': args.gamma} if args.gamma is not None else {}))
    settings = replace(settings, hubs=[h * args.hub_scale for h in settings.hubs],
                       local={k: v * args.local_scale for k, v in settings.local.items()},
                       head=settings.head // args.head_div)
    settings = replace(settings, writers=[w * args.direct_scale for w in settings.writers],
                       hidden=[w * args.direct_scale for w in settings.hidden],
                       embedding=settings.embedding * args.emb_scale)
    if args.alloc:
        settings = replace(settings, **json.loads(args.alloc))
    if args.flat_alloc:
        nrep = settings.nrep
        flat = lambda xs: [max(nrep, round(sum(xs) / len(xs) / nrep) * nrep)] * len(xs)
        settings = replace(settings, writers=flat(settings.writers), hidden=flat(settings.hidden))
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + 1))
    train = train[order]

    model = build(args.arch, config, device, args.seed, n_self=args.n_self)
    if args.init:  # start from a trained checkpoint (e.g. a backprop model, to see what Dust training does to it)
        model.load_state_dict(torch.load(Path(args.init) / 'step3000.pt', map_location=device, weights_only=True))
    if args.moe:
        from yoco import moe
        E, k, h, hs = (int(v) for v in args.moe.split(','))
        for m_ in moe.install(model, E, k, h, hs, args.moe_layers):
            m_.rate = args.moe_balance_rate
            m_.cap = args.router_cap
            m_.dense = args.moe_dense
            m_.grouped = args.moe_grouped
        if args.method != 'dust':
            model.requires_grad_(False)  # build() leaves params frozen; the bp branch re-enables them below
    optimizer = make_optimizer(model, settings)
    if args.opt == 'adamw':
        router = [p_ for n_, p_ in model.named_parameters() if 'mlp.router' in n_]
        rest = [p_ for n_, p_ in model.named_parameters() if 'mlp.router' not in n_]
        groups_ = [dict(params=rest)] + ([dict(params=router, lr=args.adam_lr * args.router_lr_mult,
                                               weight_decay=args.router_wd)] if router else [])
        optimizer = torch.optim.AdamW(groups_, lr=args.adam_lr, betas=(0.9, args.adam_beta2), eps=args.adam_eps,
                                      weight_decay=0.0)
    if args.opt == 'adam_tensor':
        optimizer = TensorAdam(model.parameters(), lr=args.adam_lr, betas=(0.9, args.adam_beta2), eps=args.adam_eps)
    if args.moe and args.moe_lr_mult != 1.0:
        expert = {id(p_) for n, p_ in model.named_parameters() if n.endswith(('w_fc', 'w_proj'))}
        groups = []
        for g in optimizer.param_groups:
            a = [p_ for p_ in g['params'] if id(p_) not in expert]
            b = [p_ for p_ in g['params'] if id(p_) in expert]
            groups += [dict(params=a, lr=g['lr'])] if a else []
            groups += [dict(params=b, lr=g['lr'] * args.moe_lr_mult)] if b else []
        optimizer = torch.optim.SGD(groups, momentum=settings.momentum)
    if args.quant != 'none':
        from yoco import quant
        quant.install(model, args.quant)
    estimator = None
    if args.method == 'dust':
        estimator = DustX(model, settings, args.seed, head=args.head, local=args.local,
                          handoff=args.handoff, handoff_draws=args.handoff_draws, oracle=args.oracle,
                          credit=args.credit, attn_c=args.attn_c,
                          res_gamma=args.res_gamma, res_blocks=args.res_blocks)
        estimator.set_precision('tf32' if args.fp32 else args.precision)
        if args.nvfp4:
            estimator.enable_nvfp4(skip_blocks=set(args.nvfp4_skip), experts=args.nvfp4_experts)
        estimator.truncate = not args.no_truncate
        estimator.kv_credit = args.kv_credit
        estimator.pool = args.pool or (bool(args.chain) and not args.chain_own)
        estimator.chain_pool = not args.chain_own
        estimator.chain_hidden = not args.chain_hubs_only
        estimator.chain = bool(args.chain)
        estimator.readout = args.readout
        estimator.readout_warmup = args.readout_warmup
        estimator.readout_scale = args.readout_scale
        estimator.readout_mode = args.readout_mode
        estimator.anchor, estimator.anchor_scale = args.anchor, args.anchor_scale
        estimator.cv_neighbor, estimator.cv_own = args.cv_neighbor, args.cv_own
        estimator.moe_fix = args.moe_fix
        estimator.orth = tuple(args.orth)
        estimator.expert_probes = args.expert_probes
        estimator.moe_cv = args.moe_cv
        estimator.anchor_boost, estimator.anchor_warmup = args.anchor_boost, args.anchor_warmup
        estimator.chain_draws = args.chain or 256
        estimator.linear_v = args.linear_v
        estimator.cache_gamma = args.cache_gamma
        estimator.oracle_re = args.oracle_re
        estimator.local_separate = args.local_separate
        estimator.cache_separate = args.cache_separate
        estimator.freeze_attn = args.freeze_attn
        estimator.live_fc, estimator.cv = args.live_fc, args.cv
        estimator.cv_beta, estimator.cv_warmup = args.cv_beta, args.cv_warmup
        estimator.kv_credit_values_only = args.linear_v  # with linear values, only the value path is clean enough
    else:
        model.requires_grad_(True)
        if args.method == 'ed':  # noise-free DUST: backprop of the expected DUST update
            expected.install(model, settings.gamma, cross_full=args.handoff)
            if args.sim_noise:
                # Noise simulator: Dust's per-token noise at each site, sized by its draw count. A Dust estimate from
                # N draws of a d-wide site has noise of about sqrt(d / N) times the error (more with leakage, less
                # with the control variate); --sim-noise C scales that as rho = sqrt(C d / N). Not added: the head,
                # value embeddings and gates. A derivative-based measuring tool, not a method.
                S_ = getattr(model, 'n_self', config.n_layer)
                emb, loc = settings.embedding, settings.local  # already scaled by the --*-scale flags

                r2 = json.loads(Path(args.sim_cv).read_text()) if args.sim_cv else {}
                expected.FLOOR = args.sim_floor
                if args.sim_tail:
                    df_, mode_ = args.sim_tail.split(',')
                    expected.TAIL = (float(df_), mode_)

                def put(mod, draws, key=None, d_=None):
                    # key: the site whose control-variate share (1 - its variance left) applies, from a recipe run
                    if d_ is None:
                        d_ = mod.weight.shape[0] if isinstance(mod, torch.nn.Linear) else mod.weight.shape[1]
                    left = 1 - r2.get(key, 0.0) if key else 1.0
                    expected.NOISE[mod] = (args.sim_noise * d_ * left / max(draws, 1)) ** 0.5
                for i, blk in enumerate(model.transformer.h):
                    put(blk.attn.c_proj, settings.writers[i], f'transformer.h.{i}.attn.c_proj')
                    put(blk.mlp.c_proj, settings.writers[i], f'transformer.h.{i}.attn.c_proj')
                    put(blk.mlp.c_fc, settings.hidden[i], f'transformer.h.{i}.mlp.c_fc')
                    put(blk.attn.c_q, loc['q'])
                    kv_d = config.n_embd // config.n_head if args.sim_kv_head else None  # leak-free: per key, per head
                    if i < S_:
                        put(blk.attn.c_k, loc['k'], d_=kv_d)
                        put(blk.attn.c_v, loc['v'], d_=kv_d)
                put(model.transformer.wte, emb, 'transformer.wte')
                if hasattr(model.transformer, 'cache_k'):
                    put(model.transformer.cache_k, loc['k'], d_=kv_d)
                    put(model.transformer.cache_v, loc['v'], d_=kv_d)
                split_ids = {id(getattr(o, n)) for o in [model.transformer] + [b.attn for b in model.transformer.h]
                         for n in ('cache_k', 'cache_v', 'c_k', 'c_v') if hasattr(o, n)}  # noised inside _split
                for mod in list(expected.NOISE):
                    if id(mod) not in split_ids:
                        mod.register_forward_hook(expected.noisy_hook)
            if args.ed_own_layers is not None:
                for i, blk in enumerate(model.transformer.h):
                    blk.attn.ed_stream = 'own' if i in args.ed_own_layers else 'full'
    if args.output is None:
        if 'MLSWEEP_RUN_DIR' not in os.environ:
            p.error('--output is required outside mlsweep')
        args.output = Path(os.environ['MLSWEEP_RUN_DIR'])
    args.output.mkdir(parents=True, exist_ok=True)
    logger = MLSweepLogger()
    (args.output / 'config.json').write_text(json.dumps(dict(
        args={k: str(v) for k, v in vars(args).items()}, model=asdict(config), recipe=asdict(settings),
        n_params=sum(q.numel() for q in model.parameters()), code=code_version()), indent=1) + '\n')

    def save(step):
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.output / f'step{step}.pt')

    if 0 in args.save_at:
        save(0)
    log = (args.output / 'metrics.jsonl').open('w')
    start = time.monotonic()
    ema = None
    high = 0  # consecutive steps with training loss above 10
    base_lr = [g['lr'] for g in optimizer.param_groups]
    clip_state = dict(avg=None, norm=0.0, clipped=0.0, steps=0)

    def part(n_):  # which part of the model a parameter belongs to, for the per-part logging
        for key, name_ in (('mlp.router', 'router'), ('mlp.w_', 'experts'), ('.mlp.', 'mlp'), ('.attn.', 'attn'),
                           ('cache_', 'cache'), ('wte', 'embedding'), ('value_embeds', 'value_embeds'),
                           ('lm_head', 'head')):
            if key in n_:
                return name_
        return 'other'
    parts = [(part(n_), p_) for n_, p_ in model.named_parameters()]
    if args.clip:
        # Wrap the optimizer's step so every path (Dust, microbatches, backprop) is clipped the same way. The limit is
        # CLIP times an average of recent (clipped) gradient norms; the first 20 steps only build the average.
        # Why adaptive rather than the standard fixed max-norm: the size of Dust's gradient estimate varies a lot
        # between recipes and model sizes, so a fixed threshold would need tuning per setup. A relative limit works
        # unchanged everywhere (the idea is not new; AutoClip, Seetharaman et al. 2020, clips at a percentile of
        # recent norms).
        raw_step = optimizer.step

        def clipped_step(*a, **kw):
            grads = [p_.grad for p_ in model.parameters() if p_.grad is not None]
            norm = torch.sqrt(sum(g.float().square().sum() for g in grads)).item()
            sq = {}
            for k_, p_ in parts:
                if p_.grad is not None:
                    sq[k_] = sq.get(k_, 0.0) + p_.grad.float().square().sum()
            clip_state['parts'] = {k_: round(v_.sqrt().item(), 4) for k_, v_ in sq.items()}
            st = clip_state
            st['steps'] += 1
            limit = args.clip * st['avg'] if st['avg'] is not None else math.inf
            st['norm'], st['clipped'] = norm, 0.0
            if st['steps'] > 20 and norm > limit:
                for g in grads:
                    g.mul_(limit / norm)
                st['clipped'] = 1.0
            kept = min(norm, limit)
            st['avg'] = kept if st['avg'] is None else 0.95 * st['avg'] + 0.05 * kept
            return raw_step(*a, **kw)
        optimizer.step = clipped_step
    wema = {n: p_.detach().float().clone() for n, p_ in model.named_parameters()} if args.ema else None

    @torch.no_grad()
    def eval_ema():
        live = {n: p_.detach().clone() for n, p_ in model.named_parameters()}
        for n, p_ in model.named_parameters():
            p_.copy_(wema[n].to(p_.dtype))
        v = evaluate(model, val)
        for n, p_ in model.named_parameters():
            p_.copy_(live[n])
        return v
    # Speed diagnostics. DUST_TIME=N: time steps N .. N+4 per estimator phase (a sync around each call), print, exit.
    # DUST_PROFILE=N: torch profiler on step N only (GPU kernels only, to keep host memory small), print, exit.
    time_at, prof_at = int(os.environ.get('DUST_TIME', '0')), int(os.environ.get('DUST_PROFILE', '0'))
    plain_forward = model.forward
    if args.compile and estimator is not None:
        # Compile the model forward the reruns use (Dust's hooks make one graph per set of perturbed sites) and the
        # head draws' elementwise chain. About 1.85x faster per step at 64 sequences on the RTX 5090, same results.
        import torch._dynamo as dynamo_
        from yoco import dustx as dustx_
        dynamo_.config.cache_size_limit = 256
        dynamo_.config.accumulated_cache_size_limit = 4096
        eager_forward = model.forward
        if not args.moe or args.moe_dense or args.moe_grouped:  # the sparse MoE path has data-dependent shapes: it would recompile every step
            model.forward = torch.compile(model.forward, dynamic=False,
                                          backend=args.compile_backend)
        dustx_.COMPILE = True
        if model.forward is not eager_forward:
            # Self-check: torch 2.11's compiled forward gives wrong per-token losses on reruns from block 0 (about 2e-2
            # off, even with a zero perturbation; reruns from later blocks and torch 2.10 are fine). Compare one rerun
            # from block 0 and one from block 1 against eager; if they disagree, rerun eagerly.
            xc, yc = split(train[:args.microbatch_size or args.batch_seqs].to(device))
            with torch.no_grad():
                estimator.capture(xc, yc)
                worst, diffs_ = 0.0, []
                moes_ = [m_ for m_ in model.modules() if hasattr(m_, 'last_ids')] if args.moe else []
                flips_ = []
                # MoE: routing is a discrete choice, so tiny rounding differences between the compiled and eager
                # forwards flip a few near-tied tokens to another expert (logged as the share of flipped slots).
                # The check itself compares the arithmetic with the routing pinned to the clean pass's.
                for pin_ in ((False, True) if moes_ else (False,)):
                    estimator.pin_routing = pin_
                    for site_, start_ in (('transformer.h.0.attn.c_proj', 0), ('transformer.h.1.attn.c_proj', 1)):
                        estimator.jitter = {site_: torch.zeros_like(estimator.outputs[site_]).repeat(2, 1, 1)}
                        estimator._start = start_
                        xr, yr = xc.repeat(2, 1), yc.repeat(2, 1)
                        got = estimator.losses(xr, yr)
                        ids_c = [m_.last_ids.clone() for m_ in moes_ if m_.last_ids.shape[0] == xr.shape[0]]
                        model.forward, compiled_forward = eager_forward, model.forward
                        ref = estimator.losses(xr, yr)
                        ids_e = [m_.last_ids.clone() for m_ in moes_ if m_.last_ids.shape[0] == xr.shape[0]]
                        model.forward = compiled_forward
                        if pin_ or not moes_:
                            worst = max(worst, (got - ref).abs().max().item())
                            diffs_.append((got - ref).abs().flatten().float())
                        else:
                            unp_ = (got - ref).abs().max().item()
                            flips_ += [(a_.sort(-1).values != b_.sort(-1).values).any(-1).float().mean().item()
                                       for a_, b_ in zip(ids_c, ids_e)]
                            print(f'compile self-check, routing free (start {start_}): max per-token loss difference '
                                  f'{unp_:.2e}; share of tokens routed differently per MoE layer '
                                  f'{[round(f_, 4) for f_ in flips_[-len(ids_c):]]}', flush=True)
                estimator.jitter, estimator._start, estimator.pin_routing = {}, 0, False
            diffs_ = torch.cat(diffs_)
            print(f'compile self-check: max per-token loss difference {worst:.2e}, mean {diffs_.mean().item():.2e}, '
                  f'share over 1e-3 {(diffs_ > 1e-3).float().mean().item():.4f}', flush=True)
            if worst > 1e-3 and args.compile_force:
                print('compile self-check failed, kept compiled (--compile-force)', flush=True)
            elif worst > 1e-3:
                print('compile self-check failed: the model forward runs eagerly (head draws stay compiled)', flush=True)
                model.forward = eager_forward
        if args.nan_debug:
            # which forward modes give NaN, eager against compiled, on one batch (diagnosing the B300 NaN)
            xd, yd = split(train[:args.microbatch_size or args.batch_seqs].to(device))
            with torch.no_grad(), estimator.autocast():
                for label_, fwd in (('eager', eager_forward), ('compiled', model.forward)):
                    rec_ = {}
                    full = fwd(xd, yd, loss_reduction='none', record=rec_).float()
                    out_ = [f'{label_}: full {full.mean().item():.4f} nan {torch.isnan(full).sum().item()}']
                    for st_ in range(1, config.n_layer):
                        tr = fwd(xd, yd, loss_reduction='none', start=st_, state=rec_).float()
                        out_.append(f'start{st_} {tr.mean().item():.4f} nan {torch.isnan(tr).sum().item()}')
                    out_.append('record: ' + ' '.join(f'x{k_} nan {torch.isnan(v_).sum().item()}'
                                                      for k_, v_ in rec_['x'].items()))
                    print('NANDEBUG', ' | '.join(out_), flush=True)
            return
    phase = {}
    if (time_at or prof_at) and estimator is not None:
        from torch.profiler import record_function
        for name_ in ('capture', 'losses', 'direct_errors', 'zo_self_local', 'zo_cross', 'head_error', 'finish', 'local_proj',
                      'readout_loss', 'cv_predict', 'cv_update', 'cv_predict_moe', 'cv_update_moe', 'cv_key'):
            if hasattr(estimator, name_):
                def wrap(fn, label):
                    def inner(*a, **kw):
                        if not time_at:
                            with record_function('dust.' + label):
                                return fn(*a, **kw)
                        torch.cuda.synchronize()
                        t0 = time.monotonic()
                        out = fn(*a, **kw)
                        torch.cuda.synchronize()
                        phase[label] = phase.get(label, 0.0) + time.monotonic() - t0
                        return out
                    return inner
                setattr(estimator, name_, wrap(getattr(estimator, name_), name_))
    prof = None
    for step in range(1, args.steps + 1):
        if time_at and step == time_at:
            torch.cuda.synchronize()
            phase.clear()
            t_prof = time.monotonic()
        if time_at and step == time_at + 5:
            torch.cuda.synchronize()
            wall = (time.monotonic() - t_prof) / 5
            print(f'TIME wall per step {wall:.3f}s  peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB', flush=True)
            for k_, v_ in sorted(phase.items(), key=lambda kv: -kv[1]):
                print(f'TIME {k_:16s} {v_ / 5:.3f}s/step  {100 * v_ / 5 / wall:.0f}%', flush=True)
            return
        if prof_at and step == prof_at:
            from torch.profiler import ProfilerActivity, profile
            torch.cuda.synchronize()
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if os.environ.get('DUST_PROFILE_CPU')
                           else [ProfilerActivity.CUDA])
            prof.__enter__()
        if prof is not None and step == prof_at + 1:
            torch.cuda.synchronize()
            prof.__exit__(None, None, None)
            ka = prof.key_averages()
            print(ka.table(sort_by=os.environ.get('DUST_PROFILE_SORT', 'cuda_time_total'), row_limit=40, max_name_column_width=70), flush=True)
            if os.environ.get('DUST_PROFILE_CPU'):
                for e_ in sorted([e_ for e_ in ka if e_.key.startswith('dust.')], key=lambda e_: -e_.device_time_total):
                    print(f'PHASE {e_.key:24s} gpu {e_.device_time_total / 1e3:9.1f} ms  cpu {e_.cpu_time_total / 1e3:9.1f} ms  calls {e_.count}', flush=True)
            return
        if args.wsd:
            f = min(1.0, (args.steps - step + 1) / (args.wsd * args.steps))
            for g, lr in zip(optimizer.param_groups, base_lr):
                g['lr'] = lr * f
        if args.draw_ramp and estimator is not None:
            u = (step - 1) / args.steps
            mult = 0.75 if u < 0.6 else 0.75 + 1.25 * (u - 0.6) / 0.4
            k = settings.nrep
            r_ = lambda v: max(k, int(round(v * mult / k)) * k)
            estimator.settings = replace(settings, writers=[r_(v) for v in settings.writers],
                                         hidden=[r_(v) for v in settings.hidden], hubs=[r_(v) for v in settings.hubs],
                                         embedding=r_(settings.embedding),
                                         local={kk: r_(v) for kk, v in settings.local.items()})
        # diagnostic steps: every 10 for the first 300, then at every evaluation
        diag = (step <= 300 and step % 10 == 0) or step == 1 or step % args.eval_every == 0 or step == args.steps \
            or (args.diag_window and args.diag_window[0] <= step <= args.diag_window[1])
        before = {n_: p_.detach().clone() for n_, p_ in model.named_parameters()} if diag else None
        flop_ctx = None
        if args.count_flops and step == 2:  # step 1 includes compilation and warm-ups; count the next one, uncompiled
            from yoco import flops as flops_
            from yoco import dustx as dustx_c
            flop_restore = (model.forward, dustx_c.COMPILE)
            model.forward, dustx_c.COMPILE = plain_forward, False
            flop_ctx = flops_.FlopCounterMode(display=False, depth=4)
            flop_ctx.__enter__()
        if estimator is not None and args.noise_probe_every and (step % args.noise_probe_every == 0 or (
                args.diag_window and args.diag_window[0] <= step <= args.diag_window[1])):
            estimator.noise_probe = True  # measured on this step's first microbatch, inside the estimator
        start_ = (step - 1) * args.batch_seqs % len(train)  # wrap around: another pass over the data
        batch = train[start_:start_ + args.batch_seqs]
        if len(batch) < args.batch_seqs:
            batch = torch.cat([batch, train[:args.batch_seqs - len(batch)]])
        batch = batch.to(device)
        x, y = split(batch)
        if estimator is not None:
            try:
                if args.microbatch_size and args.microbatch_size < len(x):
                    estimator.accum, estimator.accum_valid, losses_ = {}, 0, []
                    for xc, yc in zip(x.split(args.microbatch_size), y.split(args.microbatch_size)):
                        losses_.append(estimator.step(xc.contiguous(), yc.contiguous(), optimizer))
                    grads_, valid_ = estimator.accum, estimator.accum_valid
                    estimator.accum = None
                    optimizer.zero_grad(set_to_none=True)
                    for n_, p_ in model.named_parameters():
                        g_ = (grads_[n_] / valid_).to(p_.dtype)
                        if not torch.isfinite(g_).all():
                            raise FloatingPointError(f'Non-finite estimate for {n_}')
                        p_.grad = g_
                    optimizer.step()
                    loss = sum(losses_) / len(losses_)
                else:
                    loss = estimator.step(x, y, optimizer)
                if args.quant == 'sr':
                    quant.requantize(model, stochastic=True)
            except FloatingPointError as e_:
                print(f'step {step}: {e_}', flush=True)
                loss = math.nan
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                l = model(x, y)
            l.backward()
            optimizer.step()
            loss = l.item()
        if flop_ctx is not None:
            flop_ctx.__exit__(None, None, None)
            ntok = int(x.numel())
            fwd = flops_.forward_flops(model, x[:args.microbatch_size or len(x)], y[:args.microbatch_size or len(y)])
            fwd_tok = fwd / int(x[:args.microbatch_size or len(x)].numel())
            total = flop_ctx.get_total_flops()
            flop_rec = dict(flops_per_token=total / ntok, flops_forward_per_token=fwd_tok,
                            flops_ratio=total / ntok / fwd_tok)
            (args.output / 'flops.json').write_text(json.dumps(dict(
                step=step, tokens=ntok, total=total, forward_per_token=fwd_tok, **flop_rec,
                breakdown=flops_._breakdown(flop_ctx)), indent=1) + '\n')
            print('flops: ' + json.dumps(flop_rec), flush=True)
            model.forward, dustx_c.COMPILE = flop_restore
        else:
            flop_rec = None
        if args.moe:
            moe_load = moe.load_stats(model)
            # routing diagnostics: every 10 steps for the first 300 (router collapse under AdamW starts before step
            # 100), then at every evaluation
            if diag:
                router_ = moe.router_stats(model)
            else:
                router_ = None
            moe.balance(model)
        rec = dict(step=step, train_loss=loss, seconds=round(time.monotonic() - start, 1))
        if args.clip:
            rec['grad_norm'], rec['clipped'] = clip_state['norm'], clip_state['clipped']
            rec['grad_norm_part'] = clip_state.get('parts', {})
        if args.moe:
            rec['router_rms'] = [round(float(m_.logit_rms), 3) for m_ in model.modules() if hasattr(m_, 'logit_rms')]
            if router_ is not None:
                rec['router'] = {k_: round(v_, 4) for k_, v_ in router_.items()}
        if flop_rec:
            rec.update(flop_rec)
        if estimator is not None and getattr(estimator, 'probe_result', None):
            rec.update(estimator.probe_result)
            estimator.probe_result = None
        if step == 1:
            rec['compiled'] = float(model.forward is not plain_forward)  # 0 if compile was off or the self-check failed
        if diag:
            rec.update(diagnostics(model, parts, before, optimizer, estimator, part))
        ema = loss if ema is None else 0.9 * ema + 0.1 * loss
        high = high + 1 if loss > 10 else 0
        # diverged: a non-finite loss, the loss above 10 for 5 steps in a row (a single spike may recover), or the
        # smoothed loss above 9 after step 50
        if not math.isfinite(loss) or high >= 5 or (step > 50 and ema > 9):
            rec['diverged'] = True
            log.write(json.dumps(rec) + '\n')
            print(json.dumps(rec), flush=True)
            logger.log({'diverged': 1.0, 'train_loss': loss if math.isfinite(loss) else 99.0}, step=step)
            break
        if wema is not None and math.isfinite(loss):
            with torch.no_grad():
                for n, p_ in model.named_parameters():
                    wema[n].lerp_(p_.detach().float(), 1 - args.ema)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            rec['val_loss'] = evaluate(model, val)
            if args.halves_every and 'tsrepeat' in str(args.data_dir):
                rec['val_first_half'], rec['val_second_half'] = halves(model, val)
            if args.moe:
                rec['moe_load'] = moe_load
            if estimator is not None and estimator.cv_stats:
                # share of reward variance the memory removed, per draw group
                rec['cv_explained'] = {k: round(1 - float(b) / float(a), 3) for k, (a, b) in estimator.cv_stats.items()
                                       if float(a) > 0}
                estimator.cv_stats = {}
            if wema is not None and step >= 500:
                rec['val_loss_ema'] = eval_ema()
            print(json.dumps(rec), flush=True)
        if step in args.save_at or (args.save_every and step % args.save_every == 0 and step != args.steps):
            save(step)
        log.write(json.dumps(rec) + '\n')
        log.flush()
        logger.log(scalars(rec), step=step)
    if args.steps not in args.save_at:
        save(args.steps)
    if 'tsrepeat' in str(args.data_dir) and not rec.get('diverged') and not (args.halves_every and 'val_second_half' in rec):
        first, second = halves(model, val)
        logger.log({'val_first_half': first, 'val_second_half': second}, step=rec['step'])
        print(json.dumps(dict(val_first_half=first, val_second_half=second)), flush=True)
    logger.close()


def code_version():
    """The code a run used: the git commit and whether the tree had changes (when .git is there), and a hash of the
    Python sources, which is there either way (mlsweep may upload the code without .git)."""
    import hashlib
    import subprocess
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha1()
    for f in sorted(list(root.glob('*.py')) + list((root / 'yoco').glob('*.py'))):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    out = dict(sources_sha1=h.hexdigest()[:12])
    try:
        out['git'] = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=root, capture_output=True, text=True,
                                    timeout=10).stdout.strip() or None
        out['git_dirty'] = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root,
                                               capture_output=True, text=True, timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return out


@torch.no_grad()
def diagnostics(model, parts, before, optimizer, estimator, part):
    """Cheap per-part numbers for post-hoc debugging: weight norm, this step's update relative to it, AdamW's mean
    |m|/sqrt(v) (bias-corrected; near 1 means full-size steps on a consistent signal, near 0 mostly noise), and the
    largest clean-pass activation at Dust's sites (fp16 overflows at 65504)."""
    wn, dn, adam, cnt = {}, {}, {}, {}
    state = getattr(optimizer, 'state', {})
    group_of = {id(q): g for g in getattr(optimizer, 'param_groups', []) for q in g['params']}
    for (k_, p_), (n_, _) in zip(parts, model.named_parameters()):
        w = p_.detach().float()
        wn[k_] = wn.get(k_, 0.0) + w.square().sum()
        dn[k_] = dn.get(k_, 0.0) + (w - before[n_].float()).square().sum()
        st = state.get(p_, {})
        if 'exp_avg' in st and 'exp_avg_sq' in st:
            g = group_of.get(id(p_), {})
            b1, b2 = g.get('betas', (0.9, 0.999))
            t = float(st['step']) if 'step' in st else 1.0
            m = st['exp_avg'].float() / (1 - b1 ** t)
            v = st['exp_avg_sq'].float() / (1 - b2 ** t)
            adam[k_] = adam.get(k_, 0.0) + (m.abs() / (v.sqrt() + g.get('eps', 1e-8))).sum()
            cnt[k_] = cnt.get(k_, 0) + p_.numel()
    out = {'weight_norm': {k_: round(float(v_.sqrt()), 4) for k_, v_ in wn.items()},
           'update_ratio': {k_: float(dn[k_].sqrt() / wn[k_].sqrt().clamp(min=1e-12)) for k_ in wn}}
    if adam:
        out['adam_snr'] = {k_: round(float(adam[k_]) / cnt[k_], 4) for k_ in adam}
    if estimator is not None and getattr(estimator, 'outputs', None):
        peak = {}
        for site, o in estimator.outputs.items():
            if torch.is_tensor(o) and o.is_floating_point() and o.dim() >= 2:
                k_ = part(site)
                peak[k_] = max(peak.get(k_, 0.0), float(o.detach().abs().max()))
        out['act_peak'] = {k_: round(v_, 1) for k_, v_ in peak.items()}
    return out


def scalars(rec):
    """The numbers in a training record that mlsweep can plot (nested dicts flattened, lists summarized)."""
    out = {}
    for k, v in rec.items():
        if k == 'step':
            continue
        if isinstance(v, dict):
            out.update({f'{k}/{kk}': float(vv) for kk, vv in v.items()})
        elif isinstance(v, list):
            out[f'{k}_max'] = float(max(v))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    return out


@torch.no_grad()
@torch.no_grad()
def halves(model, rows, n=64):
    """TinyStories-repeat: mean loss on the first half of each row (plain LM) and the copyable second half."""
    L = []
    for batch in rows[:n].split(32):
        x, y = split(batch)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            L.append(model(x, y, loss_reduction='none').float().view(len(batch), -1))
    L = torch.cat(L)
    h = L.shape[1] // 2
    return L[:, :h - 1].mean().item(), L[:, h:].mean().item()


if __name__ == '__main__':
    main()
