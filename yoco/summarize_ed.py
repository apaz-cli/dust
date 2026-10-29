"""Mean and sd over seeds (42, 7, 11) for backprop and noise-free arms. Seed-42 runs have no suffix."""
import json, statistics as st
from pathlib import Path

def val(run):
    p = Path(run) / 'metrics.jsonl'
    if not p.exists():
        return None
    last = [json.loads(l) for l in open(p)][-1]
    return last.get('val_loss')

out = {}
for L, pre in (('L4', 'runs/ed/'), ('L8', 'runs/ed/L8-')):
    for lr in ('0.1', '0.2'):
        for arm in ('bp-gpt', 'bp-yoco', 'ed-gpt-g0.98', 'ed-gpt-g1', 'ed-yoco-g0.98', 'ed-yoco-g1', 'ed-yoco-g1-handoff'):
            runs = []
            for seed in ('', '-s7', '-s11'):
                if arm.startswith('bp') and seed == '':
                    a = arm.split('-')[1]
                    r = f'runs/bpsweep/{a}-{lr}' if L == 'L4' else f'runs/bpsweep/L8-{a}-{lr}'
                else:
                    r = f'{pre}{arm}-lr{lr}{seed}'
                runs.append(val(r))
            v = [x for x in runs if x is not None]
            if v:
                out[(L, lr, arm)] = (st.mean(v), st.stdev(v) if len(v) > 1 else 0, len(v))
                print(f'{L} lr{lr} {arm:22s} {st.mean(v):.3f} +- {st.stdev(v) if len(v) > 1 else 0:.3f}  n={len(v)}  {["%.3f" % x for x in v]}')
json.dump({' '.join(k): v for k, v in out.items()}, open('runs/ed_summary.json', 'w'), indent=1)
