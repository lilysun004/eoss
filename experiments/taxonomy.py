"""Blind taxonomy test: DHB + RMSProp (analysis/TAXONOMY_PREREG.md). Data-only assembly."""
import os, sys, json, time, argparse, subprocess
import numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import experiments.kspec_ladder as KL
OUT = os.path.join(_REPO, "results", "kspec_tax"); os.makedirs(OUT, exist_ok=True)
PRE = os.path.join(OUT, "preflight"); os.makedirs(PRE, exist_ok=True)
KL.PRE, KL.PRE_JSON = PRE, os.path.join(OUT, "preflight.json")
MD = os.path.join(_REPO, "TAXONOMY_RESULTS.md")

#   preflighted de-novo cells: name, optn, beta, batch, lr0, max_steps, u0_at, probe
BASE = [("dhb_b8",   "SGD-Momentum-Damped", 0.90,    8, 0.010, 30000, 8000, 3000),
        ("dhb_b16",  "SGD-Momentum-Damped", 0.90,   16, 0.010, 30000, 8000, 3000),
        ("dhb_b2048","SGD-Momentum-Damped", 0.90, 2048, 0.050, 16000, 4000, 1500),
        ("rms_b8",   "RMSProp",             0.00,    8, 0.001, 30000, 8000, 3000),
        ("rms_b16",  "RMSProp",             0.00,   16, 0.001, 30000, 8000, 3000),
        ("rms_b2048","RMSProp",             0.00, 2048, 0.001, 16000, 4000, 1500)]
#   probe cells reuse the b8 validated lr: name, optn, beta, batch, lr_key, max_steps, u0, extra
PROBES = [("dhb_kick_b8", "SGD-Momentum-Damped", 0.90, 8, "dhb_b8", 24000, 8000, ["--kick_from","12000"]),
          ("dhb_swap_b8", "SGD-Momentum-Damped", 0.90, 8, "dhb_b8", 18000, 8000, ["--batch2","2048","--switch_at","12000"]),
          ("rms_kick_b8", "RMSProp",             0.00, 8, "rms_b8", 24000, 8000, ["--kick_from","12000"]),
          ("rms_swap_b8", "RMSProp",             0.00, 8, "rms_b8", 18000, 8000, ["--batch2","2048","--switch_at","12000"])]

def tag_of(n): return f"TX_{n}_s0"
def is_done(tag):
    mp = os.path.join(OUT, tag, "meta.json")
    return os.path.exists(mp) and json.load(open(mp)).get("status") in ("done","diverged")

def _launch(tag,o,b,bb,lr,ms,u0,extra=()):
    cmd=[sys.executable,"-m","experiments.slow_sweep","--tag",tag,"--optn",o,"--beta",str(b),
         "--batch",str(bb),"--lr",str(lr),"--out_dir",OUT,"--catapult_target",str(10**9),
         "--max_steps",str(ms),"--warmup",str(10**9),"--stride","1","--u0_at",str(u0),"--seed","0"]+list(extra)
    lf=open(os.path.join(OUT,tag+".log"),"a")
    return subprocess.Popen(cmd,stdout=lf,stderr=lf)

def run_auto(conc):
    KL.preflight(BASE)
    lrs={n:(v.get("lr") if isinstance(v,dict) else v) for n,v in json.load(open(KL.PRE_JSON)).items()}
    todo=[(tag_of(n),o,b,bb,lrs.get(n),ms,u0,()) for (n,o,b,bb,_l,ms,u0,_p) in BASE if lrs.get(n)]
    todo+=[(tag_of(n),o,b,bb,lrs.get(k),ms,u0,tuple(extra)) for (n,o,b,bb,k,ms,u0,extra) in PROBES if lrs.get(k)]
    todo=[t for t in todo if not is_done(t[0])]
    print(f"[run] {len(todo)} cells (conc {conc})", flush=True)
    procs=[]
    while todo or procs:
        procs=[p for p in procs if p.poll() is None]
        while todo and len(procs)<conc:
            t=todo.pop(0); print(f"[run] launch {t[0]} lr={t[4]}", flush=True)
            procs.append(_launch(*t))
        time.sleep(20)
    print("[run] all cells complete", flush=True)

def assemble():
    import datetime
    L=[f"# TAXONOMY_RESULTS.md — blind taxonomy test, DHB + RMSProp ({datetime.date.today()})\n",
       "Predictions & rules: analysis/TAXONOMY_PREREG.md (committed before runs). DATA ONLY.\n",
       "| cell | lr | κ_B late | κ_full | gap | GBS | drift | status | notes |","|"+"---|"*9]
    denovo={}
    for grp in (BASE,[(n,o,b,bb,None,ms,u0,None) for (n,o,b,bb,_k,ms,u0,_e) in PROBES]):
        for row in grp:
            n,o,b,bb=row[0],row[1],row[2],row[3]; ms,u0=row[5],row[6]
            tag=tag_of(n); d=os.path.join(OUT,tag)
            if not os.path.exists(os.path.join(d,"dense.npz")):
                L.append(f"| {tag} | | | | | | | CENSORED | no run |"); continue
            z=np.load(os.path.join(d,"dense.npz")); meta=json.load(open(os.path.join(d,"meta.json")))
            lr=meta["lr"]; k=lr*z["lam_batch"]; dx=z["dxu"]/z["su"]
            ok=np.isfinite(k)&(np.abs(dx-1)<=0.05); idx=np.where(ok)[0]
            if len(idx)<500:
                L.append(f"| {tag} | {lr} | | | | | | {meta.get('status')} | <500 healthy steps |"); continue
            h=idx[len(idx)//2:]
            kB=float(np.nanmedian(k[h])); k1=float(np.nanmedian(k[idx[len(idx)//4:len(idx)//2]]))
            key="lam_full_w" if (o=="RMSProp" and "lam_full_w" in z.files and np.isfinite(z["lam_full_w"]).any()) else "lam_full"
            ls=z["lf_step"]; fm=(ls>=z["step"][idx[len(idx)//2]])&np.isfinite(z[key])
            kf=float(np.median(z[key][fm])) if fm.sum()>=3 else float("nan")
            gbs=float(np.nanmedian(z["gbs"][h]))
            notes=[]
            if "kick" in n and len(z.get("kick_step",[]))>0:
                notes.append(f"kicks={len(z['kick_step'])}" + (" DIED-IN-TRAIN" if meta.get("diverged") else " survived-all"))
            if "swap" in n:
                pre_lf=float(np.nanmedian(z[key][(ls>=8000)&(ls<12000)]/1)) if True else float("nan")
                post=float(np.nanmedian(z["lam_batch"][(z["step"]>=17000)]))
                notes.append(f"pre_lf(lr·λ)={pre_lf:.3f} post_lam={post:.0f}")
            denovo[n]=kB/lr if np.isfinite(kB) else float("nan")
            L.append(f"| {tag} | {lr} | {kB:.3f} | {kf if np.isfinite(kf) else float('nan'):.3f} | {kB-kf:.3f} | {gbs:.2f} | "
                     f"{kB/k1-1:+.2f} | {meta.get('status')}{'/div' if meta.get('diverged') else ''} | {'; '.join(notes)} |")
    open(MD,"w").write("\n".join(L)+"\n"); print(f"[assemble] -> {MD}", flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--auto",action="store_true")
    ap.add_argument("--assemble",action="store_true"); ap.add_argument("--concurrency",type=int,default=3)
    a=ap.parse_args()
    if a.auto: run_auto(a.concurrency); assemble()
    elif a.assemble: assemble()
