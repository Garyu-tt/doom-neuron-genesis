"""R16: simplified two-choice credit-assignment diagnostic, not a Doom result.
Fixed sensor/motor mapping; no trained readout, no game, no reward stimulation.
Compare true reward with a permutation of exactly the same reward signs.
Each trial starts with fresh fast neural state; only synaptic weights persist.
"""
import argparse,json
from pathlib import Path
import numpy as np
from hq200k_backend import HQ200K,ENCODING
from genesis_neural_interface import GROUPS
from benchmarks.hq200k_causal_probe import reset

PATTERNS=(ENCODING[:4],ENCODING[4:])
MOTOR=(GROUPS[1][1],GROUPS[2][1])

def make(seed,mode):
    return HQ200K(seed,plasticity=mode!="frozen",excitation_gain=4,
                  inhibition_gain=8,adaptation_strength=1,
                  plasticity_mode="budgeted" if mode=="budgeted" else "legacy")

def sample(b,label,noise):
    reset(b,noise)
    b.adaptation.fill(0)
    for _ in range(10):
        b._tick({})
    scores=np.zeros(2,np.int64)
    for t in range(10):
        o=b._tick({ch:2. for ch in PATTERNS[label]} if t==0 else {})
        if t>=5:
            scores+=np.array([o[list(g)].sum() for g in MOTOR])
    tie=int(scores[0]==scores[1])
    action=int(np.random.default_rng(noise ^ 0xCAFE).integers(2)) if tie else int(scores.argmax())
    return action,tie,scores.tolist()

def evaluate(b,seed):
    correct=ties=0
    confusion=np.zeros((2,2),int)
    for k in range(16):
        for label in (0,1):
            action,tie,_=sample(b,label,seed*100000+50000+k)
            correct+=action==label
            ties+=tie
            confusion[label,action]+=1
    return dict(correct=int(correct),n=32,ties=ties,confusion=confusion.tolist())

def train(b,seed,signs=None):
    rewards=[]; correct=ties=0
    # Balanced label schedule independent of network and rewards.
    labels=np.tile([0,1],32)
    np.random.default_rng(seed^12345).shuffle(labels)
    for k,label in enumerate(labels):
        action,tie,_=sample(b,int(label),seed*100000+k)
        actual=1 if action==label else -1
        sign=actual if signs is None else int(signs[k])
        correct+=actual>0
        ties+=tie
        rewards.append(sign)
        channels=[19,20,22] if sign>0 else [23,24,26]
        b.feedback("reward",channels,20,2,1,False,"credit_probe")
        # Deliberately exclude reward-induced activity; isolate weight rule.
        b.feedback_queue.clear()
    return dict(correct=int(correct),n=len(labels),ties=ties,rewards=rewards)

def main(output):
    rows=[]
    for seed in (37421,37422,37423):
        for mode in ("legacy","budgeted","frozen"):
            b=make(seed,mode)
            before=evaluate(b,seed)
            t=train(b,seed)
            after=evaluate(b,seed)
            rows.append(dict(seed=seed,mode=mode,reward="contingent",
                             before=before,training=t,after=after,l1=b.plasticity_l1))
            del b
            if mode!="frozen":
                b=make(seed,mode)
                shuffled=np.random.default_rng(seed^0xBEEF).permutation(t["rewards"])
                t=train(b,seed,shuffled)
                after=evaluate(b,seed)
                rows.append(dict(seed=seed,mode=mode,reward="permuted",
                                 before=before,training=t,after=after,l1=b.plasticity_l1))
                del b
    result=dict(scope="two-choice diagnostic, balanced labels, held-out noise, fixed action mapping, no reward pulses",
                chance=.5,rows=rows)
    output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print("HQ200K_CREDIT_RESULT",json.dumps(result,allow_nan=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args().output)
