"""Emit handbook-style SVG line charts (millimetre units, template classes) from run logs."""
import json
from pathlib import Path


def curve(run, key='val_loss'):
    pts = []
    for line in open(Path(run) / 'metrics.jsonl'):
        r = json.loads(line)
        if key in r:
            pts.append((r['step'], r[key]))
    return pts


def chart(series, x0, y0, w, h, xr, yr, xticks, yticks, xlabel='', ylabel='', fmt='{:g}'):
    """series: list of (points, css_class, style, label, (lx, ly)). Returns SVG markup.
    The plot box spans x0..x0+w, y0..y0+h (y0 at top)."""
    X = lambda v: x0 + w * (v - xr[0]) / (xr[1] - xr[0])
    Y = lambda v: y0 + h - h * (v - yr[0]) / (yr[1] - yr[0])
    o = [f'<line class="o" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>',
         f'<line class="o" x1="{x0}" y1="{y0 + h}" x2="{x0}" y2="{y0 - 1}"/>']
    for v in xticks:
        o.append(f'<line class="t" x1="{X(v):.2f}" y1="{y0 + h}" x2="{X(v):.2f}" y2="{y0 + h + 1.2}"/>')
        o.append(f'<text class="lbl-s" x="{X(v):.2f}" y="{y0 + h + 3.6}" text-anchor="middle">{fmt.format(v)}</text>')
    for v in yticks:
        o.append(f'<line class="t" x1="{x0 - 1.2}" y1="{Y(v):.2f}" x2="{x0}" y2="{Y(v):.2f}"/>')
        o.append(f'<text class="lbl-s" x="{x0 - 2}" y="{Y(v) + 0.8:.2f}" text-anchor="end">{v:g}</text>')
    if xlabel:
        o.append(f'<text class="lbl-s" x="{x0 + w / 2}" y="{y0 + h + 7.2}" text-anchor="middle">{xlabel}</text>')
    if ylabel:
        o.append(f'<text class="lbl-s" transform="translate({x0 - 11},{y0 + h / 2}) rotate(-90)" '
                 f'text-anchor="middle">{ylabel}</text>')
    for pts, cls, style, label, lpos in series:
        pts = [(a, b) for a, b in pts if xr[0] <= a <= xr[1]]
        p = ' '.join(f'{X(a):.2f},{Y(min(max(b, yr[0]), yr[1])):.2f}' for a, b in pts)
        o.append(f'<polyline class="{cls}" style="{style}" points="{p}"/>')
        if label and lpos:
            o.append(f'<text class="lbl-s" x="{lpos[0]}" y="{lpos[1]}">{label}</text>')
    return '\n'.join(o)
