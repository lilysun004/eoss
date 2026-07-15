"""
Architecture replication battery (ADDENDUM 9): mlp_l via EOSS_MODEL, tags A_*, out dir
results/kspec_arch. Strict priority order: preflight-all (lr windows do NOT transfer) ->
Tier-1 at-edge cells -> kspec tables -> sub-edge cells -> brackets (at-edge first) ->
elevation cells -> their tables/brackets. Data only; no fits.
"""
import os, sys, json, time, subprocess
import numpy as np

os.environ["EOSS_MODEL"] = "mlp_l"
os.environ["EOSS_KSPEC_OUT"] = "kspec_arch"
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
OUT = os.path.join(_REPO, "results", "kspec_arch")
os.makedirs(os.path.join(OUT, "ms"), exist_ok=True)
os.makedirs(os.path.join(OUT, "preflight"), exist_ok=True)
PY = sys.executable

#        name                optn            beta  batch  lr0     max_steps u0_at probe
CFG = [("sgd_b2048",         "SGD",          0.0, 2048, 0.0100, 10000, 3000, 1200),
       ("b2048_beta0.9",     "SGD-Momentum", 0.9, 2048, 0.0065, 10000, 3000, 1200),
       ("nest_b2048_beta0.9","SGD-Nesterov", 0.9, 2048, 0.0065, 10000, 3000, 1200),
       ("adam_b2048",        "Adam",         0.9, 2048, 0.0010, 10000, 3000, 1200),
       ("b8_beta0.9",        "SGD-Momentum", 0.9,    8, 0.0020, 20000, 6000, 2500),
       ("b32_beta0.9",       "SGD-Momentum", 0.9,   32, 0.0050, 20000, 6000, 2500),
       ("adam_b8",           "Adam",         0.9,    8, 0.0010, 20000, 6000, 2500),
       ("b8_beta0.99",       "SGD-Momentum", 0.99,   8, 0.0001, 20000, 6000, 2500),
       ("nest_b128_beta0.9", "SGD-Nesterov", 0.9,  128, 0.0060, 14000, 5000, 1500),
       ("nest_b512_beta0.9", "SGD-Nesterov", 0.9,  512, 0.0080, 10000, 3000, 1200)]
TIER1 = CFG[:4]; SUB = CFG[4:8]; ELEV = CFG[8:]
PF = os.path.join(OUT, "preflight.json")


def probe(name, optn, beta, batch, lr, steps):
    import experiments.slow_sweep as S
    tag = f"APRE_{name}_lr{lr:.6g}"
    mp = os.path.join(OUT, "preflight", tag, "meta.json")
    if not (os.path.exists(mp) and json.load(open(mp)).get("status") in ("done", "diverged")):
        S.run_cell(tag, optn, beta, batch, lr, os.path.join(OUT, "preflight"),
                   catapult_target=10**9, max_steps=steps, warmup=10**9, stride=1, seed=0)
    m = json.load(open(mp))
    if m.get("diverged"):
        return "diverged", float("nan")
    z = np.load(os.path.join(OUT, "preflight", tag, "dense.npz"))
    n = len(z["kappa"]); kap = float(np.nanmedian(z["kappa"][int(n*0.6):]))
    sn = np.log(np.maximum(z["step_norm"][int(n*0.6):], 1e-30))
    slope = float(np.polyfit(np.arange(len(sn)), sn, 1)[0]) if len(sn) > 10 else 0.0
    if kap < 0.4*2*(1-beta) or slope < -0.02:
        return "crawl", kap
    return "live", kap


def preflight():
    res = json.load(open(PF)) if os.path.exists(PF) else {}
    for (name, optn, beta, batch, lr0, *_r, psteps) in [(c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]) for c in CFG]:
        if res.get(name, {}).get("lr"):
            continue
        lr, hi, lo, acc, hist = lr0, None, None, None, []
        for it in range(6):
            t0=time.time()
            v, k = probe(name, optn, beta, batch, lr, psteps)
            hist.append(dict(lr=lr, verdict=v, kappa=None if k != k else k))
            print(f"[apre] {name} probe{it+1}: lr={lr:.6g} -> {v} (kappa {k:.3f}) [{time.time()-t0:.0f}s]", flush=True)
            if v == "live": acc = lr; break
            if v == "diverged": hi = lr; lr = float(np.sqrt(lo*hi)) if lo else lr/2
            else: lo = lr; lr = float(np.sqrt(lo*hi)) if hi else lr*1.5
        if acc is None:
            nond = [h for h in hist if h["verdict"] != "diverged"]
            acc = max(nond, key=lambda h: h["lr"])["lr"] if nond else None
            print(f"[apre] {name}: NO clean lr; best-effort {acc}", flush=True)
        res[name] = dict(lr=acc, history=hist)
        json.dump(res, open(PF, "w"), indent=1)
    print("[apre] complete:", {k: v["lr"] for k, v in res.items()}, flush=True)
    return res


def is_done(tag):
    try:
        return json.load(open(os.path.join(OUT, tag, "meta.json"))).get("status") in ("done", "diverged")
    except Exception:
        return False


def run_pool(cells, conc=3):
    procs = {}; it = iter([c for c in cells if not is_done(c[0])])
    def launch(c):
        tag, optn, beta, batch, lr, ms, u0, seed = c
        cmd = [PY, "-m", "experiments.slow_sweep", "--tag", tag, "--optn", optn,
               "--beta", str(beta), "--batch", str(batch), "--lr", str(lr), "--seed", str(seed),
               "--catapult_target", str(10**9), "--max_steps", str(ms), "--warmup", str(10**9),
               "--stride", "1", "--u0_at", str(u0), "--out_dir", OUT]
        p = subprocess.Popen(cmd, stdout=open(os.path.join(OUT, tag + ".log"), "w"),
                             stderr=subprocess.STDOUT, env=dict(os.environ))
        procs[p] = tag; print(f"  launch {tag} lr={lr:.6g}", flush=True)
    for _ in range(conc):
        try: launch(next(it))
        except StopIteration: break
    while procs:
        time.sleep(10)
        for p in list(procs):
            if p.poll() is not None:
                print(f"  finished {procs.pop(p)} rc={p.returncode}", flush=True)
                try: launch(next(it))
                except StopIteration: pass


def kspec_tables(tags):
    from experiments.kspec_estimator import analyze_cell
    for t in tags:
        d = os.path.join(OUT, t); o = os.path.join(OUT, "ms", f"{t}_kspec.json")
        if os.path.exists(o) or not os.path.exists(os.path.join(d, "dense.npz")):
            continue
        try:
            r = analyze_cell(d); json.dump(r, open(o, "w"), indent=1)
            print(f"[akspec] {t}: kspec={r.get('kappa_spec', float('nan')):.3f} "
                  f"gain={r.get('gain', float('nan')):.4f} r1={r.get('r1_dxu', float('nan')):+.2f}", flush=True)
        except Exception as e:
            print(f"[akspec] {t} ERROR {e}", flush=True)


def brackets(tags, cs=(1.05, 1.15, 1.3), steps=2000):
    import experiments.ms_cocycle as MC
    from experiments.ms_bracket import run_bracket
    done = set()
    bp = os.path.join(OUT, "ms", "bracket.json")
    if os.path.exists(bp):
        done = {(r["tag"], r["c"]) for r in json.load(open(bp))}
    for tag in tags:
        if not is_done(tag): continue
        base = "_".join(tag.split("_")[:-1])
        try:
            if not os.path.exists(os.path.join(OUT, "ms", f"{tag}_ckpt.pt")):
                MC.replay(tag, MC.CKPT_STEP[base])
            for c in cs:
                if (tag, c) not in done:
                    run_bracket(tag, c, steps=steps)
        except Exception as e:
            print(f"[abr] {tag} ERROR {e}", flush=True)


def cells_of(cfgs, pf, seeds):
    out = []
    for (name, optn, beta, batch, _l, ms, u0, _p) in cfgs:
        lr = pf[name]["lr"]
        for s in seeds:
            out.append((f"A_{name}_s{s}", optn, beta, batch, lr, ms, u0, s))
    return out


def main():
    pf = preflight()
    t1 = cells_of(TIER1, pf, [0, 1]); run_pool(t1)
    kspec_tables([c[0] for c in t1])
    sub = cells_of(SUB, pf, [0, 1]); run_pool(sub)
    kspec_tables([c[0] for c in sub])
    brackets([c[0] for c in t1])
    brackets([c[0] for c in sub], cs=(1.1, 1.25, 1.4), steps=3000)
    el = cells_of(ELEV, pf, [0]); run_pool(el)
    kspec_tables([c[0] for c in el])
    brackets([c[0] for c in el])
    print("=== ARCH BATTERY COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
