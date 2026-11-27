"""R18: preregistered replication on new graph and noise seeds.
Vary spatial concentration without inspecting graph edges or game rewards.
Offline readout, no neural training. Report instability alongside accuracy.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from hq200k_backend import HQ200K,BOUNDS,MODULE,N
from benchmarks.hq200k_causal_probe import reset
from benchmarks.hq200k_generalization import MOTOR,PATTERNS,classify
from benchmarks.hq200k_electrode_geometry import local_electrodes

def patch_electrodes(seed,patches):
    if patches==1:
        return local_electrodes(seed)
    rng=np.random.default_rng(seed ^ 0xACED)
    parts=[]
    for i,(lo,hi) in enumerate(zip(BOUNDS[:-1],BOUNDS[1:])):
        width=5 if i<4 else 4
        bases=int(lo)+MODULE*rng.choice(int(hi-lo)//MODULE,patches,replace=False)
        # Each electrode samples a balanced selection of these modules.
        chunks=[]
        for p,base in enumerate(bases):
            count=width//patches+int(p<width%patches)
            if count:
                chunks.append((base+rng.choice(MODULE,64*count,replace=False)).reshape(64,count))
        parts.append(np.concatenate(chunks,axis=1))
    result=np.concatenate(parts,axis=1)
    assert result.shape==(64,32) and np.unique(result).size==result.size
    return result

def sample(b,noise,channels,amp):
    reset(b,noise)
    b.adaptation.fill(0)
    baseline=[]
    for _ in range(30):
        b._tick({})
        baseline.append(len(b.previous)/N*100)
    obs=[]; pop=[]
    for t in range(30):
        obs.append(b._tick({ch:amp for ch in channels} if t==0 else {})[MOTOR])
        pop.append(len(b.previous)/N)
    a=np.asarray(obs)
    return a[1:5].sum(axis=0),a[5:10].sum(axis=0),max(pop),float(np.mean(baseline[-10:])),float(np.mean(pop[-10:])*100)

def run(seed,output):
    rows=[]
    for shuffled in (False,True):
        b=HQ200K(seed,plasticity=False,shuffled=shuffled,excitation_gain=4.,inhibition_gain=8.,adaptation_strength=1.)
        distributed=b.electrodes.copy()
        for patches in (0,1,2,4):
            b.electrodes=distributed if patches==0 else patch_electrodes(seed,patches)
            for amp in (1.,2.):
                x=[];y=[];peaks=[];base=[];late=[]
                for k in range(16):
                    for label,channels in enumerate(PATTERNS):
                        early,delayed,peak,baseline,tail=sample(b,seed*10000+k,channels,amp)
                        x.append([early,delayed]);y.append(label)
                        peaks.append(peak);base.append(baseline);late.append(tail)
                x=np.asarray(x);y=np.asarray(y)
                row=dict(seed=seed,shuffled=shuffled,patches=patches,amp=amp,
                         electrode_sha256=hashlib.sha256(b.electrodes.tobytes()).hexdigest(),
                         topology=b.initial_hash,test_count=32,max_peak=max(peaks),
                         mean_baseline_hz=float(np.mean(base)),max_late_hz=max(late))
                row["stability_gate"]=bool(row["max_peak"]<=.2 and row["mean_baseline_hz"]<=2 and row["max_late_hz"]<=3)
                for w,name in enumerate(("early","late")):
                    pred=classify(x[:32,w],y[:32],x[32:,w])
                    row[name+"_correct"]=int(np.count_nonzero(pred==y[32:]))
                rows.append(row)
    data=dict(scope="replication, offline linear readout; synthetic geometry",chance=.25,
              stability_gate="peak <=20%; baseline <=2 Hz; max late <=3 Hz",rows=rows)
    output.write_text(json.dumps(data,indent=2,allow_nan=False))
    print("HQ200K_GEOMETRY_VALIDATION_RESULT",json.dumps(data,allow_nan=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.seed,a.output)
