"""R05: select candidate operating points without optimizing game reward.
Frozen HQ200K, same graph and stochastic drive in each paired trial.
No game action shortcuts. Stimulus differences use only input electrodes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from hq200k_backend import HQ200K, ENCODING, N
from genesis_neural_interface import GROUPS
from benchmarks.hq200k_causal_probe import reset

GAINS=(2.5,3.,3.25,3.5,3.75,4.)
MOTOR=np.array(sorted({ch for _,channels in GROUPS[1:] for ch in channels}))
# Four balanced input patterns; these are not game actions.
PATTERNS=((), ENCODING[:4], ENCODING[4:], ENCODING)
def trial(b, seed, channels):
    reset(b,seed)
    for _ in range(20):
        b._tick({})
    trace=[]
    pop=[]
    for t in range(20):
        observed=b._tick({ch:2. for ch in channels} if t==0 else {})
        pop.append(len(b.previous))
        trace.append(observed[MOTOR])
    return np.asarray(trace),np.asarray(pop)

def main(output):
    rows=[]
    for seed in (37421,37422,37423):
        b=HQ200K(seed,plasticity=False)
        original=b.w.copy()
        for gain in GAINS:
            b.w[:]=original*gain
            for k in range(4):
                noise=seed*100+k
                trials=[trial(b,noise,p) for p in PATTERNS]
                vectors=[a[5:10].sum(axis=0) for a,_ in trials]
                # distinguishability of alternate sensory patterns at output, not just activity
                distances=[int(np.abs(vectors[i]-vectors[j]).sum())
                           for i,j in ((0,1),(0,2),(1,2),(0,3))]
                base=trials[0][1]
                rows.append(dict(seed=seed,noise_seed=noise,gain=gain,
                    distances=distances,baseline_hz=float(base.mean()/N*100),
                    baseline_peak_fraction=float(base.max()/N),
                    tail_fraction=float(trials[3][1][-5:].mean()/N),
                    stimulated_peak_fraction=float(trials[3][1].max()/N)))
        del b,original
    summary={}
    for g in GAINS:
        rs=[r for r in rows if r["gain"]==g]
        summary[str(g)]=dict(pairs=len(rs),
            distinct_patterns=sum(r["distances"][2]>0 for r in rs),
            both_vs_blank=sum(min(r["distances"][:2])>0 for r in rs),
            mean_pattern_l1=float(np.mean([r["distances"][2] for r in rs])),
            mean_baseline_hz=float(np.mean([r["baseline_hz"] for r in rs])),
            max_baseline_peak_fraction=max(r["baseline_peak_fraction"] for r in rs),
            max_stimulated_peak_fraction=max(r["stimulated_peak_fraction"] for r in rs),
            mean_tail_fraction=float(np.mean([r["tail_fraction"] for r in rs])))
    data=dict(scope="diagnostic gain sweep, no training, not CL1 calibrated",summary=summary,rows=rows)
    output.write_text(json.dumps(data,indent=2,allow_nan=False))
    print("HQ200K_GAIN_RESULT",json.dumps(data,allow_nan=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args().output)
