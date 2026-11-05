"""Train the GPT or YOCO model on TinyStories (T=512) with DUST variants or backprop.

Mirrors ~/git/dust/leakage/train_small.py: the population-256 10M recipe truncated to n_layer blocks,
3000 steps of batch 8, validation every 100 steps on the 128-row split.
"""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import GPTConfig, make_optimizer, recipe  # noqa: E402
from yoco import expected  # noqa: E402
from yoco.dustx import DustX  # noqa: E402
from yoco.model_yoco import build  # noqa: E402


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
    p.add_argument('--cv-warmup', type=int, default=20)
    p.add_argument('--hub-scale', type=int, default=1, help='multiply the o@ (attention output) draw counts')
    p.add_argument('--ed-own-layers', type=int, nargs='*', help='ed: only these layers keep own-token streams')
    p.add_argument('--n-self', type=int, help='YOCO self-decoder depth (default n_layer // 2)')
    p.add_argument('--local-separate', action='store_true', help='leak-free local k/v/gate/ve scoring (closed form)')
    p.add_argument('--oracle-re', help='regex on parameter names given their noise-free expected update')
    p.add_argument('--oracle', nargs='*', default=[], help='groups given their noise-free expected update')
    p.add_argument('--moe', help='routed MoE MLPs: "E,k,h,hs" (experts, top-k, expert hidden, shared hidden)')
    p.add_argument('--opt', choices=['sgd', 'adamw'], default='sgd', help='adamw: bp reference optimizer (lr --adam-lr)')
    p.add_argument('--adam-lr', type=float, default=3e-3)
    p.add_argument('--moe-lr-mult', type=float, default=1.0, help='lr multiplier for routed expert weights')
    p.add_argument('--moe-layers', type=int, nargs='*', help='layers that get the MoE (default all)')
    p.add_argument('--n-layer', type=int, default=4)
    p.add_argument('--n-embd', type=int, default=256)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--lr', type=float, default=0.25)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--save-at', type=int, nargs='*', default=[])
    p.add_argument('--data-dir', type=Path, default=Path('data/tinystories'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()

    device = torch.device('cuda')
    torch.set_float32_matmul_precision('high')
    heads = args.n_embd // 64
    config = GPTConfig(sequence_len=512, n_layer=args.n_layer, n_head=heads, n_kv_head=heads, n_embd=args.n_embd)
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
    train = torch.load(args.data_dir / 'train.pt', weights_only=True).long()
    val = torch.load(args.data_dir / 'validation.pt', weights_only=True).long().to(device)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + 1))
    train = train[order]

    model = build(args.arch, config, device, args.seed, n_self=args.n_self)
    if args.moe:
        from yoco import moe
        E, k, h, hs = (int(v) for v in args.moe.split(','))
        moe.install(model, E, k, h, hs, args.moe_layers)
        if args.method != 'dust':
            model.requires_grad_(False)  # build() leaves params frozen; the bp branch re-enables them below
    optimizer = make_optimizer(model, settings)
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW([p_ for p_ in model.parameters()], lr=args.adam_lr, betas=(0.9, 0.95),
                                      weight_decay=0.0)
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
        estimator.truncate = not args.no_truncate
        estimator.kv_credit = args.kv_credit
        estimator.pool = args.pool or (bool(args.chain) and not args.chain_own)
        estimator.chain_pool = not args.chain_own
        estimator.chain_hidden = not args.chain_hubs_only
        estimator.chain = bool(args.chain)
        estimator.chain_draws = args.chain or 256
        estimator.linear_v = args.linear_v
        estimator.cache_gamma = args.cache_gamma
        estimator.oracle_re = args.oracle_re
        estimator.local_separate = args.local_separate
        estimator.live_fc, estimator.cv = args.live_fc, args.cv
        estimator.cv_beta, estimator.cv_warmup = args.cv_beta, args.cv_warmup
        estimator.kv_credit_values_only = args.linear_v  # with linear values, only the value path is clean enough
    else:
        model.requires_grad_(True)
        if args.method == 'ed':  # noise-free DUST: backprop of the expected DUST update
            expected.install(model, settings.gamma, cross_full=args.handoff)
            if args.ed_own_layers is not None:
                for i, blk in enumerate(model.transformer.h):
                    blk.attn.ed_stream = 'own' if i in args.ed_own_layers else 'full'
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'config.json').write_text(json.dumps(dict(
        args={k: str(v) for k, v in vars(args).items()}, model=asdict(config), recipe=asdict(settings),
        n_params=sum(q.numel() for q in model.parameters())), indent=1) + '\n')

    def save(step):
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.output / f'step{step}.pt')

    if 0 in args.save_at:
        save(0)
    log = (args.output / 'metrics.jsonl').open('w')
    start = time.monotonic()
    ema = None
    base_lr = [g['lr'] for g in optimizer.param_groups]
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
    for step in range(1, args.steps + 1):
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
        batch = train[(step - 1) * args.batch_seqs:step * args.batch_seqs].to(device)
        x, y = split(batch)
        if estimator is not None:
            try:
                loss = estimator.step(x, y, optimizer)
                if args.quant == 'sr':
                    quant.requantize(model, stochastic=True)
            except FloatingPointError:
                loss = math.nan
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                l = model(x, y)
            l.backward()
            optimizer.step()
            loss = l.item()
        if args.moe:
            moe_load = moe.load_stats(model)
            moe.balance(model)
        rec = dict(step=step, train_loss=loss, seconds=round(time.monotonic() - start, 1))
        ema = loss if ema is None else 0.9 * ema + 0.1 * loss
        if not math.isfinite(loss) or loss > 10 or (step > 50 and ema > 9):
            rec['diverged'] = True
            log.write(json.dumps(rec) + '\n')
            print(json.dumps(rec), flush=True)
            break
        if wema is not None and math.isfinite(loss):
            with torch.no_grad():
                for n, p_ in model.named_parameters():
                    wema[n].lerp_(p_.detach().float(), 1 - args.ema)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            rec['val_loss'] = evaluate(model, val)
            if args.moe:
                rec['moe_load'] = moe_load
            if estimator is not None and estimator.cv_stats:
                # share of reward variance the memory removed, per draw group
                rec['cv_explained'] = {k: round(1 - b / a, 3) for k, (a, b) in estimator.cv_stats.items() if a > 0}
                estimator.cv_stats = {}
            if wema is not None and step >= 500:
                rec['val_loss_ema'] = eval_ema()
            print(json.dumps(rec), flush=True)
        if step in args.save_at:
            save(step)
        log.write(json.dumps(rec) + '\n')
        log.flush()
    if args.steps not in args.save_at:
        save(args.steps)


if __name__ == '__main__':
    main()
