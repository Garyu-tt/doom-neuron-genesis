"""R07: held-out readout generalization and spontaneous activity check.
Candidate gains chosen from R06, not from game reward. Separate graph seeds,
train/test noise seeds, and balanced class labels. Offline linear probe only;
it never drives a game or changes neural weights.
"""
import argparse,json
from pathlib import Path
import numpy as np
from hq200k_backend import HQ200K,ENCODING,N
from genesis_neural_interface import GROUPS
from benchmarks.hq200k_causal_probe import reset
MOTOR=np.array(sorted({c for _,cs in GROUPS[1:] for c in cs}))
PATTERNS=(ENCODING[:4],ENCODING[4:],ENCODING[::2],ENCODING[1::2])

def sample(b,noise,channels):
    reset(b,noise)
    for _ in range(30):
        b._tick({})
    observations=[]
    pop=[]
    for t in range(30):
        o=b._tick({ch:2. for ch in channels} if t==0 else {})
        observations.append(o[MOTOR])
        pop.append(len(b.previous)/N)
    return np.asarray(observations)[5:10].sum(axis=0),np.asarray(pop)

def classify(train,labels,test):
    # Standardized ridge regression to one-hot labels, fit on training only.
    mean=train.mean(axis=0); std=np.maximum(train.std(axis=0),1.)
    x=(train-mean)/std; y=np.eye(4)[labels]
    w=np.linalg.solve(x.T@x+np.eye(x.shape[1])*10,x.T@(y-.25))
    return np.argmax(((test-mean)/std)@w,axis=1)

def main(source,output):
    r=json.loads(source.read_text())["summary"]
    # Predeclared diagnostic thresholds, not clinical thresholds.
    valid=[(key,s) for key,s in r.items() if s["mean_baseline_hz"]<=2 and s["max_late_hz"]<=3
           and s["max_stimulated_peak"]<=.20 and s["distinct"]>=4]
    valid.sort(key=lambda pair:(pair[1]["both_vs_blank"],pair[1]["distinct"],pair[1]["mean_l1"]),reverse=True)
    candidates=[key for key,_ in valid[:2]]
    # Keep reference control even if no candidate meets criteria.
    rows=[]
    for seed in (48101,48102,48103):
        b=HQ200K(seed,plasticity=False)
        orig=b.w.copy()
        for key in ["1.0/1.0",*candidates]:
            e,i=map(float,key.split("/"))
            b.w[:]=orig*np.where(orig>0,e,i)
            train=[]; test=[]; yt=[]; yv=[]; peaks=[]; late=[]
            for k in range(12):
                for label,p in enumerate(PATTERNS):
                    v,pop=sample(b,seed*100+k,p)
                    if k<8:
                        train.append(v); yt.append(label)
                    else:
                        test.append(v); yv.append(label)
                    peaks.append(float(pop.max()))
                    late.append(float(pop[-10:].mean()*100))
            pred=classify(np.asarray(train),np.asarray(yt),np.asarray(test))
            truth=np.asarray(yv)
            cm=np.zeros((4,4),int)
            np.add.at(cm,(truth,pred),1)
            rows.append(dict(graph_seed=seed,gain=key,test_count=len(truth),
                             correct=int((pred==truth).sum()),confusion=cm.tolist(),
                             max_peak=max(peaks),mean_late_hz=float(np.mean(late))))
        del orig,b
    data=dict(scope="held-out noise-seed classification within each unseen graph; offline probe, not neural learning",
              candidate_rule="baseline <=2 Hz; max late <=3 Hz; peak <=20%; distinct >=4/6",
              selected=candidates,chance=.25,rows=rows)
    output.write_text(json.dumps(data,indent=2,allow_nan=False))
    print("HQ200K_GENERALIZATION_RESULT",json.dumps(data,allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ei",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    main(a.ei,a.output)
