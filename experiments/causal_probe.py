"""Causal probes at the small-batch plateau (analysis/CAUSAL_PROBE_PREREG.md). Data-only assembly."""
import os, sys, json, time, argparse, subprocess
import numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
OUT = os.path.join(_REPO, "results", "kspec_causal"); os.makedirs(OUT, exist_ok=True)
MD = os.path.join(_REPO, "CAUSAL_PROBE_RESULTS.md")

#        tag         optn            beta  b   lr      steps  u0    extra flags
CELLS = [
 ("CPU_sgd_b8up",   "SGD",          0.00,   8, 0.010, 18000, 8000, ["--batch2","2048","--switch_at","12000"]),
 ("CPU_sgdm_b8up",  "SGD-Momentum", 0.90,   8, 0.002, 18000, 8000, ["--batch2","2048","--switch_at","12000"]),
 ("CPU_adam_b8up",  "Adam",         0.90,   8, 0.001, 18000, 8000, ["--batch2","2048","--switch_at","12000"]),
 ("CPU_muon_b8up",  "Muon",         0.95,   8, 0.001, 18000, 8000, ["--batch2","2048","--switch_at","12000"]),
 ("CPD_sgdm_dn",    "SGD-Momentum", 0.90, 2048, 0.0065, 17000, 2000, ["--batch2","8","--switch_at","5000"]),
 ("CPD_adam_dn",    "Adam",         0.90, 2048, 0.001, 17000, 2000, ["--batch2","8","--switch_at","5000"]),
 ("CPD_muon_dn",    "Muon",         0.95, 2048, 0.001, 17000, 2000, ["--batch2","8","--switch_at","5000"]),
 ("CPK_sgd_b8",     "SGD",          0.00,   8, 0.010, 24000, 8000, ["--kick_from","12000"]),
 ("CPK_sgdm_b8",    "SGD-Momentum", 0.90,   8, 0.002, 24000, 8000, ["--kick_from","12000"]),
 ("CPK_nest_b8",    "SGD-Nesterov", 0.90,   8, 0.002, 24000, 8000, ["--kick_from","12000"]),
 ("CPK_adam_b8",    "Adam",         0.90,   8, 0.001, 24000, 8000, ["--kick_from","12000"]),
 ("CPK_muon_b8",    "Muon",         0.95,   8, 0.001, 24000, 8000, ["--kick_from","12000"])]

def is_done(tag):
    mp = os.path.join(OUT, tag, "meta.json")
    return os.path.exists(mp) and json.load(open(mp)).get("status") in ("done","diverged")

def run_auto(conc):
    todo = [c for c in CELLS if not is_done(c[0])]
    print(f"[run] {len(todo)} cells (conc {conc})", flush=True)
    procs=[]
    while todo or procs:
        procs=[p for p in procs if p.poll() is None]
        while todo and len(procs)<conc:
            tag,o,b,bb,lr,ms,u0,extra = todo.pop(0)
            print(f"[run] launch {tag}", flush=True)
            cmd=[sys.executable,"-m","experiments.slow_sweep","--tag",tag,"--optn",o,"--beta",str(b),
                 "--batch",str(bb),"--lr",str(lr),"--out_dir",OUT,"--catapult_target",str(10**9),
                 "--max_steps",str(ms),"--warmup",str(10**9),"--stride","1","--u0_at",str(u0),"--seed","0"]+extra
            lf=open(os.path.join(OUT,tag+".log"),"a")
            procs.append(subprocess.Popen(cmd,stdout=lf,stderr=lf))
        time.sleep(20)
    print("[run] all cells complete", flush=True)

def _med(a): return float(np.nanmedian(a)) if len(a) else float("nan")

def assemble():
    import datetime
    L=[f"# CAUSAL_PROBE_RESULTS.md — batch-swap ablations + kick trains ({datetime.date.today()})\n",
       "Frozen predictions: analysis/CAUSAL_PROBE_PREREG.md (committed before runs). DATA ONLY.\n"]
    L.append("## Swap cells (P1 up / P2 down)\n")
    L.append("| cell | swap | pre-swap λ_B med | λ_B @ +2k (%Δ) | λ_B end-4k med (drift/2k) | diverged |")
    L.append("|---|---|---|---|---|---|")
    for tag,o,b,bb,lr,ms,u0,extra in CELLS:
        if not tag.startswith(("CPU","CPD")): continue
        d=os.path.join(OUT,tag)
        if not os.path.exists(d+"/dense.npz"): L.append(f"| {tag} | | | | | CENSORED |"); continue
        z=np.load(d+"/dense.npz"); meta=json.load(open(d+"/meta.json"))
        sw=int(extra[extra.index("--switch_at")+1]); st=z['step']; lam=z['lam_batch']
        pre=_med(lam[(st>=sw-4000)&(st<sw)]); p2k=_med(lam[(st>=sw+1500)&(st<sw+2500)])
        n=len(st); e=_med(lam[st>=st[-1]-4000]); e0=_med(lam[(st>=st[-1]-8000)&(st<st[-1]-4000)])
        L.append(f"| {tag} | {bb}→{extra[1]} @ {sw} | {pre:.1f} | {p2k:.1f} ({100*(p2k/pre-1):+.0f}%) | "
                 f"{e:.1f} ({100*(e/e0-1) if e0 else float('nan'):+.0f}%) | {meta.get('diverged')} |")
    L.append("\n## Kick cells (P3): per amplitude tier, median over kicks\n")
    L.append("| cell | A0 | tier ×A0 | n | Δλ_B(+300) % vs pre-600 med | net return cumsum(dxu)/amp @600 |")
    L.append("|---|---|---|---|---|---|")
    for tag,o,b,bb,lr,ms,u0,extra in CELLS:
        if not tag.startswith("CPK"): continue
        d=os.path.join(OUT,tag)
        if not os.path.exists(d+"/dense.npz"): L.append(f"| {tag} | | | | | CENSORED |"); continue
        z=np.load(d+"/dense.npz"); st=z['step']; lam=z['lam_batch']; ks=z['kick_step']; ka=z['kick_amp']
        A0=_med(np.abs(ka))/np.median([2,8,32,128]) if len(ka) else float("nan")
        tiers={}
        for k,a in zip(ks,ka):
            i0=np.searchsorted(st,k)
            pre=_med(lam[max(0,i0-600):i0]); post=_med(lam[i0+250:i0+350])
            ret=float(np.nansum(z['dxu'][i0:i0+600])/a) if abs(a)>0 else float("nan")
            tier=min([2,8,32,128],key=lambda t:abs(abs(a)/max(A0,1e-30)-t)) if np.isfinite(A0) else 0
            tiers.setdefault(tier,[]).append((100*(post/pre-1) if pre else float("nan"),ret))
        for t in sorted(tiers):
            v=tiers[t]
            L.append(f"| {tag} | {A0:.2e} | {t} | {len(v)} | {_med([x[0] for x in v]):+.1f}% | {_med([x[1] for x in v]):+.2f} |")
    open(MD,"w").write("\n".join(L)+"\n"); print(f"[assemble] -> {MD}", flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--auto",action="store_true")
    ap.add_argument("--assemble",action="store_true"); ap.add_argument("--concurrency",type=int,default=3)
    a=ap.parse_args()
    if a.auto: run_auto(a.concurrency); assemble()
    elif a.assemble: assemble()
