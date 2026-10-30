"""Is jittering the writer pair (attn.c_proj + mlp.c_proj) in one pass worth it? Compare, at equal forward
passes, DUST's estimate of each writer's output gradient when the pair shares N draws vs. each writer gets
N/2 draws alone. The target is the exact own-token gradient dL_t/dY_t from autograd (what DUST estimates).

Prediction from measure.py: per-site error is proportional to (1 + kappa_group) / N paired and
(1 + kappa) / (N / 2) split, so pairing wins for a site only when kappa_group < 1 + 2 * kappa.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import DUST, GPTConfig, make_model, recipe  # noqa: E402
import leakage.measure as measure  # noqa: E402
from leakage.measure import Probe, per_token_losses  # noqa: E402


def own_token_grads(model, probe, x, y, sites, K=16):
    """dL_t / dY_t for every token t and site, shape [B, T, d] per site."""
    B, T = x.shape
    out = {s: None for s in sites}
    for b in range(B):
        rows = {s: [] for s in sites}
        for c in range(0, T, K):
            tk = torch.arange(c, min(c + K, T), device=x.device)
            k = len(tk)
            probe.active = True
            loss = per_token_losses(model, x[b:b + 1].expand(k, -1).contiguous(), y[b:b + 1].expand(k, -1).contiguous())
            loss.gather(1, tk[:, None]).sum().backward()
            probe.active = False
            for s in sites:
                rows[s].append(probe.leaves[s].grad[torch.arange(k), tk])
            probe.leaves.clear()
        for s in sites:
            g = torch.cat(rows[s])[None]
            out[s] = g if out[s] is None else torch.cat([out[s], g])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/tinystories/validation.pt'))
    p.add_argument('--sequences', type=int, default=4)
    p.add_argument('--draws', type=int, default=64, help='forward passes per arm (N)')
    p.add_argument('--trials', type=int, default=8)
    p.add_argument('--blocks', type=int, nargs='+', default=[0])
    p.add_argument('--sigma', type=float, default=0.2)
    p.add_argument('--fp32', action='store_true', help='run DUST draws without bf16 autocast, fp32 weights')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()

    device = torch.device('cuda')
    torch.set_float32_matmul_precision('high')
    cfg = json.loads((args.checkpoint.parent / 'config.json').read_text())
    model = make_model(GPTConfig(**cfg['model']), device, 42)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    data = torch.load(args.data, weights_only=True)[:args.sequences].long().to(device)
    x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()

    settings = recipe('10M', 256)  # only nrep (draw chunk, for reward centering) matters here
    estimator = DUST(model, settings, seed=0)
    if args.fp32:
        # DUST stores embeddings and rotary tables in bf16 and autocasts its forwards; undo all of it.
        model.float()
        model.cos, model.sin = model.cos.float(), model.sin.float()
        estimator.dtype = torch.float32
        estimator.autocast = nullcontext
        measure.AUTOCAST = False
    probe = Probe(model)
    results = {}
    for blk in args.blocks:
        A, Bs = f'transformer.h.{blk}.attn.c_proj', f'transformer.h.{blk}.mlp.c_proj'
        target = own_token_grads(model, probe, x, y, [A, Bs])
        estimator.capture(x, y)
        err = {(arm, s): [] for arm in ('paired', 'split') for s in (A, Bs)}
        for trial in range(args.trials):
            estimator.generator.manual_seed(1000 + trial)
            with torch.no_grad():
                paired = estimator.direct_errors([A, Bs], args.draws, args.sigma)
                split = {**estimator.direct_errors([A], args.draws // 2, args.sigma),
                         **estimator.direct_errors([Bs], args.draws // 2, args.sigma)}
            for arm, est in (('paired', paired), ('split', split)):
                for s in (A, Bs):
                    t = target[s]
                    # Relative squared error of the per-token output-gradient estimate, pooled over tokens.
                    err[(arm, s)].append(((est[s] - t).square().sum() / t.square().sum()).item())
        results[blk] = {f'{arm}:{s.split(".", 2)[-1]}': dict(rel_mse=sum(v) / len(v), trials=v)
                        for (arm, s), v in err.items()}
        for k, v in results[blk].items():
            print(f'block {blk} {k:24s} rel_mse={v["rel_mse"]:.3f}', flush=True)
    estimator.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(checkpoint=str(args.checkpoint), draws=args.draws, sigma=args.sigma,
                                           fp32=args.fp32,
                                           sequences=args.sequences, results=results), indent=1) + '\n')


if __name__ == '__main__':
    main()
