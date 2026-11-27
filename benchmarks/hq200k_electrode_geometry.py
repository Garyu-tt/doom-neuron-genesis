"""R17: matched electrode geometry and topology, no neural learning.
Local placement uses one randomly chosen full module per region; same cell
count and regional composition as distributed electrodes. No edge inspection.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from hq200k_backend import HQ200K,BOUNDS,MODULE,N
from benchmarks.hq200k_causal_probe import reset
from benchmarks.hq200k_generalization import MOTOR,PATTERNS,classify

def local_electrodes(seed):
    rng=np.random.default_rng(seed ^ 0xACED)
    parts=[]
    for i,(lo,hi) in enumerate(zip(BOUNDS[:-1],BOUNDS[1:])):
        width=5 if i<4 else 4
        base=int(lo)+MODULE*int(rng.integers(int(hi-lo)//MODULE))
        parts.append((base+rng.choice(MODULE,64*width,replace=False)).reshape(64,width))
    result=np.concatenate(parts,axis=1)
    assert result.shape==(64,32) and np.unique(result).size==result.size
    return result

def sample(b,noise,channels):
    reset(b,noise)
    b.adaptation.fill(0)
    for _ in range(30):
        b._tick({})
    counts=[]; peaks=[]
    for t in range(20):
        counts.append(b._tick({ch:2. for ch in channels} if t==0 else {})[MOTOR])
        peaks.append(len(b.previous)/N)
    a=np.asarray(counts)
    return a[1:5].sum(axis=0),a[5:10].sum(axis=0),max(peaks)

def run(seed,output):
    rows=[]
    for shuffled in (False,True):
        for e,i,a in ((1.,1.,0.),(4.,8.,1.)):
            b=HQ200K(seed,plasticity=False,shuffled=shuffled,
                      excitation_gain=e,inhibition_gain=i,adaptation_strength=a)
            distributed=b.electrodes.copy()
            for placement in ("distributed","local"):
                b.electrodes=distributed if placement=="distributed" else local_electrodes(seed)
                x=[]; y=[]; peaks=[]
                for k in range(12):
                    for label,channels in enumerate(PATTERNS):
                        early,late,peak=sample(b,seed*100+k,channels)
                        x.append([early,late]);y.append(label);peaks.append(peak)
                x=np.asarray(x);y=np.asarray(y)
                row=dict(seed=seed,shuffled=shuffled,placement=placement,gains=[e,i,a],
                         topology=b.initial_hash,electrode_sha256=hashlib.sha256(b.electrodes.tobytes()).hexdigest(),
                         max_peak=max(peaks),test_count=16)
                for w,name in enumerate(("early","late")):
                    pred=classify(x[:32,w],y[:32],x[32:,w])
                    row[name+"_correct"]=int(np.count_nonzero(pred==y[32:]))
                rows.append(row)
            del b
    data=dict(scope="offline held-out linear readout; local module placement is synthetic, not biological geometry",chance=.25,rows=rows)
    output.write_text(json.dumps(data,indent=2,allow_nan=False))
    print("HQ200K_GEOMETRY_RESULT",json.dumps(data,allow_nan=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.seed,a.output)
