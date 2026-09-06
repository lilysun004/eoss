"""CB2: out-of-sample confirmation of the formula-free G2 gap law (CB_LAW_PREREG.md G2 addendum)."""
import os, sys, json, time, argparse
import numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import experiments.cb_sweep as CB
import experiments.kspec_ladder as KL
from experiments.kspec_estimator import _welch_transfer

CELLS = [("hb085_b8",   "SGD-Momentum", 0.85,  8, 0.002, 30000, 8000, 3000),
         ("hb06_b16",   "SGD-Momentum", 0.60, 16, 0.005, 30000, 8000, 3000),
         ("nest06_b16", "SGD-Nesterov", 0.60, 16, 0.005, 30000, 8000, 3000),
         ("muon05_b8",  "Muon",         0.50,  8, 0.001, 30000, 8000, 3000)]
MD = os.path.join(_REPO, "G2_CONFIRM_RESULTS.md")
def tag_of(n): return f"CB2_{n}_s0"

def run_auto(conc):
    KL.preflight(CELLS)
    pj = json.load(open(KL.PRE_JSON))
    lrs = {n: (v.get("lr") if isinstance(v, dict) else v) for n, v in pj.items()}
    todo = [(n,o,b,bb,lrs.get(n),ms,u0) for (n,o,b,bb,_l,ms,u0,_p) in CELLS if lrs.get(n)]
    todo = [t for t in todo if not os.path.exists(os.path.join(CB.OUT, tag_of(t[0]), "meta.json"))
            or json.load(open(os.path.join(CB.OUT, tag_of(t[0]), "meta.json"))).get("status") not in ("done","diverged")]
    print(f"[run] {len(todo)} cells (conc {conc})", flush=True)
    procs=[]
    while todo or procs:
        procs=[p for p in procs if p.poll() is None]
        while todo and len(procs)<conc:
            n,o,b,bb,lr,ms,u0=todo.pop(0)
            print(f"[run] launch {tag_of(n)} lr={lr}", flush=True)
            # reuse CB.launch but with CB2 tag: temporary shim
            import subprocess
            cmd=[sys.executable,"-m","experiments.slow_sweep","--tag",tag_of(n),"--optn",o,"--beta",str(b),
                 "--batch",str(bb),"--lr",str(lr),"--out_dir",CB.OUT,"--catapult_target",str(10**9),
                 "--max_steps",str(ms),"--warmup",str(10**9),"--stride","1","--u0_at",str(u0),"--seed","0"]
            lf=open(os.path.join(CB.OUT,tag_of(n)+".log"),"a")
            procs.append(subprocess.Popen(cmd,stdout=lf,stderr=lf))
        time.sleep(20)
    print("[run] all cells complete", flush=True)

def assemble():
    import datetime
    L=[f"# G2_CONFIRM_RESULTS.md — CB2 out-of-sample cells for the formula-free gap law ({datetime.date.today()})\n",
       "Frozen protocol + predictions: CB_LAW_PREREG.md G2 addendum (committed before runs). DATA ONLY.\n",
       "| cell | β/mom | b | lr | κ_B | κ_full | gap | G2 | **gap×G2** | drift | GBS |","|"+"---|"*11]
    for (n,optn,beta,batch,_l,_ms,_u0,_p) in CELLS:
        d=os.path.join(CB.OUT,tag_of(n))
        if not os.path.exists(os.path.join(d,"dense.npz")):
            L.append(f"| {tag_of(n)} | | | | | | | | | | CENSORED |"); continue
        z=np.load(os.path.join(d,"dense.npz")); meta=json.load(open(os.path.join(d,"meta.json"))); lr=meta["lr"]
        k=lr*z["lam_batch"]; dx=z["dxu"]/z["su"]
        ok=np.isfinite(k)&(np.abs(dx-1)<=0.05); idx=np.where(ok)[0]; h=idx[len(idx)//2:]
        kB=float(np.nanmedian(k[h])); k1=float(np.nanmedian(k[idx[len(idx)//4:len(idx)//2]]))
        ls=z["lf_step"]; fm=(ls>=z["step"][idx[len(idx)//2]])&np.isfinite(z["lam_full"])
        kf=float(np.median(z["lam_full"][fm]))
        m=np.isfinite(z["gu0"])&np.isfinite(z["su0"]); g,s=z["gu0"][m],z["su0"][m]
        nper=int(min(2048,2**np.floor(np.log2(max(len(g)//6,256)))))
        w,T,Pgg,coh=_welch_transfer(g,s,nper); nb=max(3,len(w)//64)
        G2=float(np.average(T[1:nb],weights=Pgg[1:nb])/lr)
        L.append(f"| {tag_of(n)} | {beta} | {batch} | {lr} | {kB:.3f} | {kf:.3f} | {kB-kf:.3f} | {G2:.2f} | "
                 f"**{(kB-kf)*G2:.2f}** | {kB/k1-1:+.2f} | {float(np.nanmedian(z['gbs'][h])):.2f} |")
    open(MD,"w").write("\n".join(L)+"\n"); print(f"[assemble] -> {MD}", flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--auto",action="store_true")
    ap.add_argument("--assemble",action="store_true"); ap.add_argument("--concurrency",type=int,default=3)
    a=ap.parse_args()
    if a.auto: run_auto(a.concurrency); assemble()
    elif a.assemble: assemble()
