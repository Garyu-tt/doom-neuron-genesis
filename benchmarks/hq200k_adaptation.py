"""R10: local spike-triggered adaptation, paired stimulus trials.
No change to graph, no labels/reward inside neural dynamics.
"""
import argparse,json
from pathlib import Path
import numpy as np
from hq200k_backend import HQ200K,N,ENCODING
from genesis_neural_interface import GROUPS
from benchmarks.hq200k_causal_probe import reset
MOTOR=np.array(sorted({c for _,cs in GROUPS[1:] for c in cs}))
CONFIGS=((3.75,4.,0.),(3.75,4.,.2),(3.75,4.,.5),(4.,6.,.2),(4.,6.,.5),(4.,6.,1.),(4.,8.,.5),(4.,8.,1.))
PATTERNS=((),ENCODING[:4],ENCODING[4:])
def trial(b,noise,p):
    reset(b,noise)
    b.adaptation.fill(0)
    for _ in range(100): b._tick({})
    outputs=[]; pop=[]
    for t in range(100):
        o=b._tick({ch:2. for ch in p} if t==0 else {})
        outputs.append(o[MOTOR])
        pop.append(len(b.previous)/N)
    return np.array(outputs),np.array(pop)
def main(output):
    rows=[]
    for seed in (37421,37422,37423):
        for e,i,a in CONFIGS:
            b=HQ200K(seed,plasticity=False,excitation_gain=e,inhibition_gain=i,adaptation_strength=a)
            for k in range(2):
                data=[trial(b,seed*100+k,p) for p in PATTERNS]
                v=[x[0][5:10].sum(axis=0) for x in data]
                d=[int(np.abs(v[x]-v[y]).sum()) for x,y in ((0,1),(0,2),(1,2))]
                rows.append(dict(seed=seed,noise=seed*100+k,exc=e,inh=i,adapt=a,distances=d,
                    baseline_hz=float(data[0][1].mean()*100),
                    peak=max(float(x[1].max()) for x in data),
                    max_late_hz=max(float(x[1][-50:].mean()*100) for x in data)))
    summary={}
    for e,i,a in CONFIGS:
        rs=[r for r in rows if (r["exc"],r["inh"],r["adapt"])==(e,i,a)]
        summary[f"{e}/{i}/{a}"]=dict(trials=len(rs),distinct=sum(r["distances"][2]>0 for r in rs),
            both_vs_blank=sum(min(r["distances"][:2])>0 for r in rs),
            mean_baseline_hz=float(np.mean([r["baseline_hz"] for r in rs])),
            max_late_hz=max(r["max_late_hz"] for r in rs),max_peak=max(r["peak"] for r in rs))
    result=dict(scope="adaptation calibration, not learning",summary=summary,rows=rows)
    output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print("HQ200K_ADAPT_RESULT",json.dumps(result,allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args().output)
