"""
GOLD sweep: ONE run per (optimizer x batch) cell on mlp_s, seed 0, stride 1, with kappa_spec AND GBS
computed from the SAME dense.npz over the SAME analysis window [u0_at, end]. Produces GOLD_RESULTS.md
+ kspec_results/gold/{gold_table.csv, <cell>_kspec.json}, superseding every earlier kappa_spec/GBS
table (which mixed seeds, budgets and eras).

Grid: {SGD, SGD-Momentum b0.9, SGD-Nesterov b0.9, Adam b1=0.9, Muon mom=0.95} x {8,32,128,512,2048}.
Per-batch protocol == kspec_ladder.LADDER exactly (max_steps / u0_at / probe).
Priority order: all b2048 cells first, then b512, b128, b32, b8 (partial completion -> top items).

Standing rules honoured: liveness-bisect preflight on EVERY cell (kspec_ladder.preflight re-used with
this sweep's dirs); raw signed primitives only (slow_sweep runner); results-dir-per-job
(results/kspec_gold); NO fitting -- the only statistic is the pre-registered secondary test
(KSPEC_PREREG_ANNOTATIONS.md: per-cell (GBS_med, kappa_spec), Pearson corr, origin-anchored slope).
The estimator is experiments.kspec_estimator (grep-certified formula-free), untouched.

Usage:
  python -m experiments.gold_sweep --auto [--concurrency 3]   # preflight+run per priority group, then assemble
  python -m experiments.gold_sweep --preflight | --run | --assemble | --status
  python -m experiments.gold_sweep --smoke                      # assemble path on an existing cell dir
"""
import os, sys, json, time, argparse, subprocess, threading, queue, csv, datetime
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import experiments.kspec_ladder as KL
import experiments.kspec_estimator as KE

OUT = os.path.join(_REPO, "results", "kspec_gold")
PRE = os.path.join(OUT, "preflight")
PRE_JSON = os.path.join(OUT, "preflight.json")
RES = os.path.join(_REPO, "kspec_results", "gold")
MD = os.path.join(_REPO, "GOLD_RESULTS.md")
for d in (OUT, PRE, RES):
    os.makedirs(d, exist_ok=True)
# point the ladder helpers at THIS sweep's directories (they read module globals at call time)
KL.PRE, KL.PRE_JSON, KL.OUT = PRE, PRE_JSON, OUT

# per-batch protocol == kspec_ladder.LADDER (max_steps, u0_at, probe)
PROTO = {8: (30000, 8000, 3000), 32: (30000, 8000, 3000), 128: (20000, 5000, 2000),
         512: (16000, 4000, 1500), 2048: (16000, 4000, 1500)}
BATCHES = [2048, 512, 128, 32, 8]                       # priority order
#        key     optn            beta
OPTS = [("sgd",  "SGD",          0.00),
        ("sgdm", "SGD-Momentum", 0.90),
        ("nest", "SGD-Nesterov", 0.90),
        ("adam", "Adam",         0.90),
        ("muon", "Muon",         0.95)]                  # Muon momentum 0.95 == prior L_/A_muon cells
_OLD_PRE = os.path.join(_REPO, "results", "kspec", "preflight.json")
_OLD_KEY = {"sgdm": "b{b}_beta0.9", "nest": "nest_b{b}_beta0.9", "adam": "adam_b{b}"}


def _lr0(key, b):
    """lr0 for the bisect: validated mlp_s lr where the cell exists, else a seed lr."""
    old = json.load(open(_OLD_PRE)) if os.path.exists(_OLD_PRE) else {}
    k = _OLD_KEY.get(key, "").format(b=b)
    if k and old.get(k, {}).get("lr"):
        return float(old[k]["lr"])
    if key == "sgd":  return 0.01
    if key == "muon": return 0.001                       # L_muon_b2048 lr; bisect walks from here
    if key == "adam": return 0.001
    return 0.005


def cells_for(batches=BATCHES):
    out = []
    for b in batches:
        ms, u0, pr = PROTO[b]
        for key, optn, beta in OPTS:
            out.append((f"{key}_b{b}", optn, beta, b, _lr0(key, b), ms, u0, pr))
    return out


def tag_of(name): return f"G_{name}_s0"


def is_done(tag):
    try:
        return json.load(open(os.path.join(OUT, tag, "meta.json"))).get("status") in ("done", "diverged")
    except Exception:
        return False


def launch(name, optn, beta, batch, max_steps, u0_at):
    lr = json.load(open(PRE_JSON))[name]["lr"]
    assert lr, f"{name}: no validated lr"
    tag = tag_of(name)
    cmd = [sys.executable, "-m", "experiments.slow_sweep", "--tag", tag, "--optn", optn,
           "--beta", str(beta), "--batch", str(batch), "--lr", str(lr), "--seed", "0",
           "--catapult_target", str(10**9), "--max_steps", str(max_steps), "--warmup", str(10**9),
           "--stride", "1", "--u0_at", str(u0_at), "--out_dir", OUT]
    p = subprocess.Popen(cmd, stdout=open(os.path.join(OUT, tag + ".log"), "w"), stderr=subprocess.STDOUT)
    print(f"  launch {tag} lr={lr:.6g} pid={p.pid}", flush=True)
    return p


class Pool:
    """Bounded-concurrency launcher fed by a queue so preflight of the next group overlaps runs."""
    def __init__(self, conc):
        self.q, self.conc, self.procs, self.done = queue.Queue(), conc, {}, threading.Event()
        self.t = threading.Thread(target=self._loop, daemon=True); self.t.start()
    def submit(self, cell): self.q.put(cell)
    def close(self): self.q.put(None)
    def _loop(self):
        closed = False
        while True:
            for p in list(self.procs):
                if p.poll() is not None:
                    c = self.procs.pop(p); print(f"  finished {tag_of(c[0])} rc={p.returncode}", flush=True)
            while not closed and len(self.procs) < self.conc:
                try: c = self.q.get(timeout=1)
                except queue.Empty: break
                if c is None: closed = True; break
                if is_done(tag_of(c[0])):
                    print(f"  skip {tag_of(c[0])} (done)", flush=True); continue
                self.procs[launch(c[0], c[1], c[2], c[3], c[5], c[6])] = c
            if closed and not self.procs: break
            time.sleep(10)
        self.done.set()
    def wait(self): self.done.wait()


def run_auto(conc):
    pool = Pool(conc)
    for b in BATCHES:                                    # priority groups
        grp = cells_for([b])
        print(f"[auto] preflight group b{b}", flush=True)
        KL.preflight(grp)                                # serial bisect, resume-safe, records PRE_JSON
        for c in grp:
            if json.load(open(PRE_JSON)).get(c[0], {}).get("lr"): pool.submit(c)
            else: print(f"[auto] {c[0]}: NO lr -> cell CENSORED (no run)", flush=True)
    pool.close(); pool.wait()
    print("[run] all cells complete", flush=True)


def run_only(conc):
    pool = Pool(conc)
    for c in cells_for():
        if json.load(open(PRE_JSON)).get(c[0], {}).get("lr"): pool.submit(c)
    pool.close(); pool.wait(); print("[run] all cells complete", flush=True)


# ---------------------------------------------------------------- assemble (data only, no fitting)
HEALTH_TOL = 0.05   # |dxu/su - 1| <= tol: applied step along u equals intended step (float32 fidelity; optimizer-agnostic)
HEALTH_RUN = 200    # sustained death = first index i with >= HEALTH_RUN/2 unhealthy steps in [i, i+HEALTH_RUN)
MIN_HEALTHY = 512   # fewer healthy prefix steps than this -> cell censored (listed, not dropped)


def health(z):
    """Per-step numerical-health mask over the fixed-frame window + first sustained-death step.
    Pre-registered 2026-08-30 (before any gold cell finished) after the heavy-ball b2048 probe
    (analysis/HB_B2048_GBS_PROBE.md): at loss ~1e-9 the float32 update falls below half-ulp, the applied
    step is a fraction of the intended one, and per-step GBS is rounding noise. Nothing optimizer-specific."""
    win = np.isfinite(z["gu0"]) & np.isfinite(z["gu"]) & np.isfinite(z["su"]) & np.isfinite(z["lam_batch"])
    idx = np.where(win)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = z["dxu"][idx] / z["su"][idx]
    ok = np.abs(ratio - 1) <= HEALTH_TOL
    ok &= np.isfinite(ratio)
    unh = (~ok).astype(float)
    death_step = None
    if len(unh) >= HEALTH_RUN:
        c = np.convolve(unh, np.ones(HEALTH_RUN), "valid")
        bad = np.where(c >= HEALTH_RUN / 2)[0]
        if len(bad):
            death_step = int(z["step"][idx[int(bad[0])]])
    hmask = np.zeros(len(win), bool); hmask[idx[ok]] = True
    if death_step is not None:
        hmask &= z["step"] < death_step
    return win, hmask, death_step, float(ok.mean()) if len(ok) else float("nan")


def _gbs_stats(g):
    if g.size == 0 or not np.isfinite(g).any():
        return dict(gbs_n=0, gbs_med=float("nan"), gbs_q25=float("nan"), gbs_q75=float("nan"), gbs_mean=float("nan"))
    return dict(gbs_n=int(np.isfinite(g).sum()), gbs_med=float(np.nanmedian(g)), gbs_q25=float(np.nanpercentile(g, 25)),
                gbs_q75=float(np.nanpercentile(g, 75)), gbs_mean=float(np.nanmean(g)))


def analyze(cell_dir):
    """kappa_spec via the certified estimator + GBS stats. PRIMARY numbers: healthy prefix
    [u0_at, death_step) with per-step health mask for GBS. Full-window numbers kept as *_fullwin."""
    z = np.load(os.path.join(cell_dir, "dense.npz"))
    win, hmask, death_step, healthy_frac = health(z)
    r_full = KE.analyze_cell(cell_dir)
    r = KE.analyze_cell(cell_dir, end_step=death_step) if death_step is not None else dict(r_full)
    n_healthy = int(hmask.sum())
    r.update(death_step=death_step, healthy_frac=healthy_frac, n_healthy=n_healthy,
             kappa_spec_fullwin=r_full.get("kappa_spec", float("nan")), n_window_full=r_full.get("n_window"))
    r.update(_gbs_stats(z["gbs"][hmask]))
    r["gbs_med_fullwin"] = float(np.nanmedian(z["gbs"][win])) if win.sum() else float("nan")
    if not r.get("ok"):
        r.update(cell=os.path.basename(cell_dir), why=f"numerically dead before window: death_step={death_step}, "
                 f"healthy prefix {n_healthy} < {MIN_HEALTHY} ({r.get('why')})")
        for k in ("optn", "beta", "batch", "lr", "kappa_raw", "kappa_spec", "r1_dxu", "gain", "kappa_drift", "stationary"):
            r.setdefault(k, r_full.get(k))
    h1, h2 = r.get("kappa_spec_h1"), r.get("kappa_spec_h2")
    r["halves_agree"] = bool(h1 and h2 and np.isfinite(h1) and np.isfinite(h2) and abs(h2 / h1 - 1) < 0.25)
    k = r.get("kappa_spec", float("nan"))
    r["gbs_over_kspec"] = r["gbs_med"] / k if (r.get("ok") and k) else float("nan")
    m = json.load(open(os.path.join(cell_dir, "meta.json")))
    r.update(steps_measured=m.get("steps"), elapsed_s=m.get("elapsed_s"), tag=m.get("tag"))
    return r


COLS = ["cell", "optn", "beta", "batch", "lr", "steps_measured", "status", "diverged", "n_window",
        "kappa_raw", "r1_dxu", "omega_star_over_pi", "gain", "kappa_spec", "kappa_spec_ci_lo",
        "kappa_spec_ci_hi", "kappa_drift", "stationary", "res_limited", "kappa_spec_fixed",
        "kappa_spec_h1", "kappa_spec_h2", "gbs_med", "gbs_q25", "gbs_q75", "gbs_mean", "gbs_n",
        "gbs_over_kspec", "death_step", "healthy_frac", "n_healthy", "kappa_spec_fullwin", "gbs_med_fullwin",
        "halves_agree", "ok", "why"]


def _f(x, nd=3):
    try:
        return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"
    except Exception:
        return str(x)


def _agreement(rows):
    """Pre-registered secondary test: corr + origin-anchored slope of GBS_med on kappa_spec."""
    def stat(rs):
        k = np.array([r["kappa_spec"] for r in rs]); g = np.array([r["gbs_med"] for r in rs])
        m = np.isfinite(k) & np.isfinite(g); k, g = k[m], g[m]
        if len(k) < 3: return dict(n=int(len(k)), corr=float("nan"), slope=float("nan"))
        return dict(n=int(len(k)), corr=float(np.corrcoef(k, g)[0, 1]), slope=float(g @ k / (k @ k)))
    ok = [r for r in rows if r.get("ok")]
    return dict(all=stat(ok), excl_heavyball=stat([r for r in ok if r["optn"] != "SGD-Momentum"]),
                stationary_only=stat([r for r in ok if r.get("stationary")]))


def assemble(cell_dirs, res_dir=RES, md_path=MD, expected=None, preflight=None):
    rows = []
    for d in cell_dirs:
        try:
            r = analyze(d)
        except Exception as e:
            r = dict(cell=os.path.basename(d), ok=False, why=f"analyze error: {e!r}")
        rows.append(r)
        json.dump(r, open(os.path.join(res_dir, os.path.basename(d) + "_kspec.json"), "w"), indent=1)
    with open(os.path.join(res_dir, "gold_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({c: r.get(c) for c in COLS})
    agr = _agreement(rows)
    json.dump(agr, open(os.path.join(res_dir, "agreement.json"), "w"), indent=1)
    # ---- markdown
    have = {r["cell"] for r in rows}
    missing = [tag_of(c[0]) for c in (expected or []) if tag_of(c[0]) not in have]
    L = []
    L.append(f"# GOLD_RESULTS.md — kappa_spec AND GBS on the same run ({datetime.date.today()})\n")
    L.append("**These numbers SUPERSEDE every earlier kappa_spec / GBS table** (KSPEC_RESULTS.md, "
             "RESULTS_MASTER.md §1/§3, SUMMARY.md, HANDOFF.md, kspec_results/*.json). Earlier tables mixed "
             "seeds, budgets and runner eras; here each cell is ONE run and both quantities are read from "
             "the SAME `dense.npz` over the SAME analysis window.\n")
    L.append("**Analysis window (pre-registered 2026-08-30 11:15, before any gold cell finished).** Primary "
             "numbers use the *healthy prefix* [u0_at, death_step): a step is healthy iff |dxu/su − 1| ≤ 0.05 "
             "(the applied step along the top mode equals the intended step — a float32-fidelity check, no "
             "optimizer knowledge); death_step = first sustained failure (≥100 unhealthy in 200 steps). "
             "Motivation: `analysis/HB_B2048_GBS_PROBE.md` — heavy-ball b2048 reaches loss ~1e-9 by step ~6000, "
             "the update drops below float32 half-ulp, and per-step GBS becomes rounding noise (0.33) while "
             "κ_spec's PSD-weighted integral hides it. Full-window numbers are kept in the table "
             "(`κ_spec full / GBS full`) so nothing is hidden; cells with < 512 healthy steps are censored, "
             "listed, never dropped.\n")
    L.append("**Protocol.** mlp_s (789k), CIFAR-10 num_data=2048, MSE, CPU, seed 0, stride 1 everywhere. "
             "Optimizers: SGD, SGD-Momentum β=0.9, SGD-Nesterov β=0.9, Adam β₁=0.9, Muon momentum=0.95. "
             "Batches 8/32/128/512/2048 with max_steps/u0_at = 30000/8000, 30000/8000, 20000/5000, "
             "16000/4000, 16000/4000 (== kspec_ladder.LADDER). lr per cell from a liveness-bisect preflight "
             "(`results/kspec_gold/preflight.json`; live = not diverged, κ_late ≥ 0.4·2(1−β), step-norm "
             "slope > −0.02). Estimator: `experiments/kspec_estimator.py` (formula-free, grep-certified): "
             "κ_spec = median(λ_B)·∫|T̂(ω)| dμ(ω), T̂ = Welch S_gu,su/P_gu, μ = P_gu; GBS = sᵀH_Bs/(−gᵀs) per "
             "step (slow_sweep), median/IQR/mean over the identical mask. **No fitting, no interpretation** "
             "— the only statistic is the pre-registered secondary test (KSPEC_PREREG_ANNOTATIONS.md).\n")
    hdr = ("| cell | lr | steps | κ_raw | r₁ | ω*/π | gain | **κ_spec** [CI] | drift | stat. | "
           "**GBS_med** [IQR] | GBS/κ_spec | κ_spec full / GBS full | death step (healthy %) | flags |")
    sep = "|" + "---|" * 15
    def row(r):
        if not r.get("ok"):
            return (f"| {r['cell']} | | | | | | | | | | | | {_f(r.get('kappa_spec_fullwin'))} / {_f(r.get('gbs_med_fullwin'))} | "
                    f"{r.get('death_step')} ({_f(100*r.get('healthy_frac', float('nan')),0)}%) | **CENSORED: {r.get('why')}** |")
        fl = []
        if r.get("diverged"): fl.append("diverged")
        if r.get("res_limited"): fl.append("ω-res-limited")
        if not r.get("stationary"): fl.append("nonstationary")
        if not r.get("halves_agree"): fl.append("halves-disagree")
        return (f"| {r['cell']} | {_f(r['lr'],5)} | {r.get('steps_measured')} | {_f(r['kappa_raw'])} | "
                f"{_f(r['r1_dxu'],2)} | {_f(r['omega_star_over_pi'],2)} | {_f(r['gain'],4)} | "
                f"**{_f(r['kappa_spec'])}** [{_f(r['kappa_spec_ci_lo'])},{_f(r['kappa_spec_ci_hi'])}] | "
                f"{_f(r['kappa_drift'],3)} | {'✓' if r['stationary'] else '✗'} | "
                f"**{_f(r['gbs_med'])}** [{_f(r['gbs_q25'])},{_f(r['gbs_q75'])}] | "
                f"{_f(r['gbs_over_kspec'])} | {_f(r.get('kappa_spec_fullwin'))} / {_f(r.get('gbs_med_fullwin'))} | "
                f"{r.get('death_step') if r.get('death_step') is not None else '—'} ({_f(100*r.get('healthy_frac', float('nan')),0)}%) | "
                f"{', '.join(fl)} |")
    order = {o[0]: i for i, o in enumerate(OPTS)}
    def key(r):
        b = r.get("batch") or 0
        k = r["cell"].replace("G_", "").split("_b")[0]
        return (-int(b), order.get(k, 9), r["cell"])
    rows_s = sorted(rows, key=key)
    L.append("## 1. Full table (25 cells, priority order b2048 → b8)\n")
    L += [hdr, sep] + [row(r) for r in rows_s]
    if missing:
        L.append("\n**Cells with no run (censored — preflight found no lr, or run not finished):** "
                 + ", ".join(missing))
    L.append("\n## 2. b2048 cells only (the disputed heavy-ball GBS-vs-κ_spec cell lives here)\n")
    L += [hdr, sep] + [row(r) for r in rows_s if r.get("ok") and r.get("batch") == 2048]
    L.append("\n## 3. Pre-registered secondary test: GBS_med vs κ_spec agreement (report only)\n")
    L.append("Registered prediction: slope ≈ 1 through the origin, both instruments = 2 at any binding "
             "edge and short by the same factor on sub-marginal cells.\n")
    L.append("| subset | n | Pearson corr | origin slope (GBS on κ_spec) |")
    L.append("|---|---|---|---|")
    for k, v in agr.items():
        L.append(f"| {k} | {v['n']} | {_f(v['corr'])} | {_f(v['slope'])} |")
    if preflight:
        L.append("\n## 4. Preflight (liveness bisect) record\n")
        L.append("| cell | accepted lr | clean | probes |")
        L.append("|---|---|---|---|")
        for k, v in preflight.items():
            L.append(f"| {k} | {v.get('lr')} | {v.get('clean')} | "
                     + "; ".join(f"{h['lr']:.5g}→{h['verdict']}" for h in v.get("history", [])) + " |")
    L.append("\n## 5. Files\n")
    L.append("- per-cell JSON: `kspec_results/gold/<cell>_kspec.json`; table: `kspec_results/gold/gold_table.csv`; "
             "agreement: `kspec_results/gold/agreement.json`\n- raw runs: `results/kspec_gold/<cell>/dense.npz` "
             "(signed primitives gu/su/mu/dxu/gu0/su0/gbs/kappa/lam_batch/a_t …)\n- driver: `experiments/gold_sweep.py`")
    open(md_path, "w").write("\n".join(L) + "\n")
    print(f"[assemble] {len(rows)} cells -> {md_path}; agreement {json.dumps(agr)}", flush=True)
    return rows


def status():
    pf = json.load(open(PRE_JSON)) if os.path.exists(PRE_JSON) else {}
    for c in cells_for():
        tag = tag_of(c[0]); mp = os.path.join(OUT, tag, "meta.json")
        st = "pending"
        if os.path.exists(mp):
            m = json.load(open(mp)); st = f"{m.get('status')} steps={m.get('steps')}/{c[5]}"
        print(f"{tag:22s} lr={pf.get(c[0], {}).get('lr')}  {st}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("auto", "preflight", "run", "assemble", "status", "smoke"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--concurrency", type=int, default=3)
    a = ap.parse_args()
    if a.smoke:
        sd = os.path.join(_REPO, "results", "kspec", "L_b2048_beta0.9_s0")
        tmp = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "gold_smoke"); os.makedirs(tmp, exist_ok=True)
        rows = assemble([sd], res_dir=tmp, md_path=os.path.join(tmp, "GOLD_SMOKE.md"), expected=cells_for([2048]))
        print(json.dumps({k: rows[0].get(k) for k in ("kappa_spec", "gbs_med", "gbs_q25", "gbs_q75", "gbs_over_kspec", "n_window", "gbs_n")}, indent=1))
    if a.preflight: KL.preflight(cells_for())
    if a.auto: run_auto(a.concurrency)
    if a.run: run_only(a.concurrency)
    if a.assemble or a.auto:
        dirs = [os.path.join(OUT, tag_of(c[0])) for c in cells_for()
                if os.path.exists(os.path.join(OUT, tag_of(c[0]), "dense.npz"))]
        pf = json.load(open(PRE_JSON)) if os.path.exists(PRE_JSON) else {}
        assemble(dirs, expected=cells_for(), preflight=pf)
    if a.status: status()
