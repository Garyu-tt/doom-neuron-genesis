"""R20: decodability versus long-window dynamical stability.
No topology changes, no learning, no game. Candidate grid fixed before run.
"""
import argparse,json
from pathlib import Path
import numpy as np
from hq200k_backend import HQ200K,N
from benchmarks.hq200k_causal_probe import reset
from benchmarks.hq200k_generalization import MOTOR,PATTERNS,classify

# excitation, inhibition, threshold increment, threshold decay (ms)
CONFIGS=((4.,8.,1.,200.),(4.,8.,1.5,200.),(4.,8.,2.,200.),
         (4.,8.,1.,400.),(4.,8.,1.5,400.),(3.75,8.,1.,200.),
         (3.75,8.,1.5,200.),(3.5,8.,1.,200.))

def sample(b,seed,channels):
    reset(b,seed)
    b.adaptation.fill(0)
    base=[];pop=[];obs=[]
    for _ in range(100):
        b._tick({})
        base.append(len(b.previous)/N*100)
    for t in range(100):
        o=b._tick({ch:2. for ch in channels} if t==0 else {})
        if t<10:
            obs.append(o[MOTOR])
        pop.append(len(b.previous)/N)
    obs=np.asarray(obs)
    return obs[1:5].sum(axis=0),obs[5:10].sum(axis=0),max(pop),float(np.mean(base[-50:])),float(np.mean(pop[-50:])*100)

def run(seed,output):
    rows=[]
    for e,i,a,tau in CONFIGS:
        b=HQ200K(seed,plasticity=False,excitation_gain=e,inhibition_gain=i,
                  adaptation_strength=a,adaptation_tau_ms=tau,electrode_patches=1)
        x=[];y=[];peaks=[];base=[];tail=[]
        for k in range(12):
            for label,channels in enumerate(PATTERNS):
                early,late,p,bl,tl=sample(b,seed*10000+k,channels)
                x.append([early,late]);y.append(label)
                peaks.append(p);base.append(bl);tail.append(tl)
        x=np.asarray(x);y=np.asarray(y)
        row=dict(seed=seed,config=[e,i,a,tau],test_count=16,
                 max_peak=max(peaks),mean_baseline_hz=float(np.mean(base)),
                 max_baseline_hz=max(base),max_late_hz=max(tail),
                 topology=b.initial_hash,electrode_sha256=b.electrode_sha256)
        row["stability_gate"]=bool(row["max_peak"]<=.2 and row["max_baseline_hz"]<=2 and row["max_late_hz"]<=3)
        for w,name in enumerate(("early","late")):
            pred=classify(x[:32,w],y[:32],x[32:,w])
            row[name+"_correct"]=int(np.count_nonzero(pred==y[32:]))
        rows.append(row)
        print("R20_CONFIG",json.dumps(row),flush=True)
        del b
    data=dict(scope="one second baseline and one second after single stimulus; offline held-out readout",
              chance=.25,stability_gate="peak <=20%; worst baseline <=2 Hz; worst late <=3 Hz",
              rows=rows)
    output.write_text(json.dumps(data,indent=2,allow_nan=False))
    print("HQ200K_STABLE_SENSORY_RESULT",json.dumps(data,allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.seed,a.output)
