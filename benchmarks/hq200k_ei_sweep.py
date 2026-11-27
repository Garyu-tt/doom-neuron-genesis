"""R06: independently tune excitation and inhibition, without changing topology.
Frozen weights within trials, fixed common noise. A two-second baseline tests
whether a candidate continues to self-excite after stimulus decay.
"""
import argparse,json
from pathlib import Path
import numpy as np
from hq200k_backend import HQ200K,ENCODING,N
from genesis_neural_interface import GROUPS
from benchmarks.hq200k_causal_probe import reset
MOTOR=np.array(sorted({c for _,cs in GROUPS[1:] for c in cs}))
CONFIGS=((3.75,3.75),(3.85,3.85),(3.9,3.9),(4.,4.),(4.,5.),(4.,6.),(4.,8.),(4.25,6.),(4.25,8.),(4.5,8.))
PATTERNS=((),ENCODING[:4],ENCODING[4:])
def run(b,noise,channels):
    reset(b,noise)
    for _ in range(100):
        b._tick({})
    trace=[]
    pop=[]
    for t in range(100):
        obs=b._tick({ch:2. for ch in channels} if t==0 else {})
        trace.append(obs[MOTOR])
        pop.append(len(b.previous))
    return np.asarray(trace),np.asarray(pop)/N
def main(output):
    rows=[]
    for seed in (37421,37422,37423):
        b=HQ200K(seed,plasticity=False)
        original=b.w.copy()
        for exc,inh in CONFIGS:
            b.w[:]=original*np.where(original>0,exc,inh)
            for k in range(2):
                noise=seed*100+k
                results=[run(b,noise,p) for p in PATTERNS]
                vec=[x[0][5:10].sum(axis=0) for x in results]
                delta=[int(np.abs(vec[a]-vec[c]).sum()) for a,c in ((0,1),(0,2),(1,2))]
                base=results[0][1]
                late=[float(x[1][-50:].mean()*100) for x in results]
                rows.append(dict(seed=seed,noise=noise,exc=exc,inh=inh,distances=delta,
                    baseline_hz=float(base.mean()*100),baseline_peak=float(base.max()),
                    stimulated_peak=max(float(x[1].max()) for x in results[1:]),
                    late_hz=late,late_excess_hz=max(abs(x-late[0]) for x in late[1:])))
        del b,original
    summary={}
    for e,i in CONFIGS:
        rs=[r for r in rows if (r["exc"],r["inh"])==(e,i)]
        summary[f"{e}/{i}"]=dict(trials=len(rs),
            distinct=sum(r["distances"][2]>0 for r in rs),
            both_vs_blank=sum(min(r["distances"][:2])>0 for r in rs),
            mean_l1=float(np.mean([r["distances"][2] for r in rs])),
            mean_baseline_hz=float(np.mean([r["baseline_hz"] for r in rs])),
            max_baseline_peak=max(r["baseline_peak"] for r in rs),
            max_stimulated_peak=max(r["stimulated_peak"] for r in rs),
            mean_late_excess_hz=float(np.mean([r["late_excess_hz"] for r in rs])),
            max_late_hz=max(max(r["late_hz"]) for r in rs))
    data=dict(scope="synthetic excitation/inhibition calibration, no gameplay or learning",summary=summary,rows=rows)
    output.write_text(json.dumps(data,indent=2,allow_nan=False))
    print("HQ200K_EI_RESULT",json.dumps(data,allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args().output)
