"""
Aggregate the comprehensive sweep: read every results/comprehensive_sweep/<tag>/meta.json,
print a regime table (R, GBS, kappa, cos(buf,uB), perturb-gamma, escape, sharpening rise,
catapult kurtosis/Hill), classify marginal vs metastable, and write summary.{json,csv}.

Usage:  python -m experiments.comprehensive_analyze
Safe to run any time while the sweep is in progress (reports only completed cells).
"""
import os, sys, json, csv, glob
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(_REPO, 'results', 'comprehensive_sweep')


def classify(m):
    R = m.get('plateau_R'); gbs = m.get('plateau_gbs')
    if R is None or gbs is None or not np.isfinite(R) or not np.isfinite(gbs):
        return '?'
    if R < 1.0 and gbs > 1.4:
        return 'MARGINAL'
    if R > 2.0 and gbs < 1.0:
        return 'metastable'
    return 'crossover'


def main():
    metas = []
    for p in sorted(glob.glob(os.path.join(OUT_ROOT, '*', 'meta.json'))):
        try:
            with open(p) as f:
                metas.append(json.load(f))
        except Exception:
            pass
    if not metas:
        print("no completed cells yet"); return
    metas.sort(key=lambda m: (m.get('beta', 0), m.get('B', 0), m.get('lr_index', 0)))

    cols = ['tag', 'B', 'beta', 'lr', 'regime', 'plateau_gbs', 'plateau_kappa', 'plateau_R',
            'plateau_cos_buf_uB', 'perturb_gamma_proj_mean', 'perturb_gamma_proj_std',
            'perturb_escape_over_natural', 'sharpen_net_rise_mean', 'sharpen_net_rise_std',
            'sharpen_interpolated', 'catapult_kurtosis', 'catapult_p99_p50', 'catapult_hill',
            'stationary', 'status']
    rows = []
    for m in metas:
        st = m.get('stationarity', {})
        rows.append(dict(
            tag=m.get('tag'), B=m.get('B'), beta=m.get('beta'), lr=m.get('lr'),
            regime=classify(m),
            plateau_gbs=m.get('plateau_gbs'), plateau_kappa=m.get('plateau_kappa'),
            plateau_R=m.get('plateau_R'), plateau_cos_buf_uB=m.get('plateau_cos_buf_uB'),
            perturb_gamma_proj_mean=m.get('perturb_gamma_proj_mean'),
            perturb_gamma_proj_std=m.get('perturb_gamma_proj_std'),
            perturb_escape_over_natural=m.get('perturb_escape_over_natural'),
            sharpen_net_rise_mean=m.get('sharpen_net_rise_mean'),
            sharpen_net_rise_std=m.get('sharpen_net_rise_std'),
            sharpen_interpolated=m.get('sharpen_interpolated'),
            catapult_kurtosis=m.get('catapult_kurtosis'), catapult_p99_p50=m.get('catapult_p99_p50'),
            catapult_hill=m.get('catapult_hill'),
            stationary=st.get('stabilized'), status=m.get('status')))

    with open(os.path.join(OUT_ROOT, 'summary.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(OUT_ROOT, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)

    def fnum(x, fmt='{:.3f}'):
        return fmt.format(x) if isinstance(x, (int, float)) and x is not None and np.isfinite(x) else '   -  '

    print(f"\n{'tag':18s} {'regime':9s} {'GBS':>6s} {'kap':>6s} {'R':>7s} {'cosBu':>6s} "
          f"{'g_proj':>8s} {'esc':>5s} {'shrp':>7s} {'interp':>6s} {'kurt':>7s} {'stat':>5s}")
    for r in rows:
        esc = r['perturb_escape_over_natural']
        print(f"{str(r['tag']):18s} {r['regime']:9s} {fnum(r['plateau_gbs'],'{:.2f}'):>6s} "
              f"{fnum(r['plateau_kappa'],'{:.2f}'):>6s} {fnum(r['plateau_R'],'{:.3f}'):>7s} "
              f"{fnum(r['plateau_cos_buf_uB'],'{:.3f}'):>6s} "
              f"{fnum(r['perturb_gamma_proj_mean'],'{:+.4f}'):>8s} "
              f"{(str(esc) if esc else '>40'):>5s} "
              f"{fnum(r['sharpen_net_rise_mean'],'{:+.3f}'):>7s} "
              f"{str(r['sharpen_interpolated'])[:5]:>6s} {fnum(r['catapult_kurtosis'],'{:.1f}'):>7s} "
              f"{str(r['stationary'])[:5]:>5s}")

    n = len(rows)
    div = sum(1 for r in rows if r['status'] == 'diverged')
    nonstat = sum(1 for r in rows if r['stationary'] is False)
    interp = sum(1 for r in rows if r['sharpen_interpolated'])
    print(f"\ncells reported: {n}   diverged: {div}   non-stationary: {nonstat}   interpolated: {interp}")
    for reg in ('MARGINAL', 'crossover', 'metastable', '?'):
        print(f"  {reg}: {sum(1 for r in rows if r['regime']==reg)}")
    print(f"\nwrote {os.path.join(OUT_ROOT,'summary.json')} and summary.csv")


if __name__ == '__main__':
    main()
