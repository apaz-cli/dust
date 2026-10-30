"""Final validation loss, EMA loss and wall time of every night-2 run -> runs/n2/summary.json."""
import json
from pathlib import Path

out = {}
for d in sorted(Path('runs/n2').iterdir()):
    m = d / 'metrics.jsonl'
    if not d.is_dir() or not m.exists():
        continue
    rows = [json.loads(l) for l in open(m)]
    vals = [r for r in rows if 'val_loss' in r]
    last = rows[-1]
    out[d.name] = dict(step=last['step'], val=vals[-1]['val_loss'] if vals else None,
                       ema=vals[-1].get('val_loss_ema') if vals else None, seconds=last['seconds'],
                       diverged=bool(last.get('diverged')))
Path('runs/n2/summary.json').write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(f"{k:18s} step {v['step']:5d}  val {v['val'] or 0:.3f}  ema {v['ema'] or 0:.3f}  {v['seconds']:7.1f} s{'  DIVERGED' if v['diverged'] else ''}")
