"""R21: isolate action-specific causal eligibility, not a biological model.
The executed action identifies its motor electrode group. Only observed
pre/post spike chains ending at that group are eligible for action-traced
updates. No correct action label enters the update, only scalar reward.
Uses R18 geometry for a short-window diagnostic; R20 stability pending.
"""
import argparse,json
from pathlib import Path
import numpy as np
from hq200k_backend import HQ200K,ENCODING,N
from genesis_neural_interface import GROUPS
from benchmarks.hq200k_causal_probe import reset

PATTERNS=(ENCODING[:4],ENCODING[4:])
MOTOR=(GROUPS[1][1],GROUPS[2][1])

def select_edges(dst,traces,terminal_cells,mode):
    """Return unique active synapse IDs with recency strengths."""
    ids=[]; strengths=[]
    frontier=np.zeros(len(dst),bool)
    for tick in range(len(traces)-1,-1,-1):
        rows,slots=traces[tick]
        if mode=="traced":
            if tick>=1:
                frontier[terminal_cells]=True
            # Each backward step follows only an observed causal connection.
            keep=frontier[dst[rows,slots]]
            rows,slots=rows[keep],slots[keep]
            frontier.fill(False)
            frontier[rows]=True
        if len(rows):
            ids.append(rows.astype(np.int64)*dst.shape[1]+slots)
            strengths.append(np.full(len(rows),.001*.8**(len(traces)-1-tick)))
    if not ids:
        return np.empty(0,np.int64),np.empty(0,np.float64)
    unique,inverse=np.unique(np.concatenate(ids),return_inverse=True)
    strength=np.zeros(len(unique))
    np.maximum.at(strength,inverse,np.concatenate(strengths))
    return unique,strength

def update(b,ids,strength,reward,budget=2.):
    if not len(ids) or not reward:
        return 0.
    rows,slots=ids//32,ids%32
    old=b.w[rows,slots]
    delta=strength*min(1.,budget/max(float(strength.sum()),1e-12))
    # A rewarded active inhibitory edge becomes less inhibitory; preserve sign.
    new=(np.sign(old)*np.clip(np.abs(old)+np.sign(old)*reward*delta,.001,.25)).astype(np.float32)
    b.w[rows,slots]=new
    return float(np.abs(new-old).sum(dtype=np.float64))

def sample(b,label,noise):
    reset(b,noise)
    b.adaptation.fill(0)
    for _ in range(10):
        b._tick({})
    scores=np.zeros(2,np.int64)
    traces=[]
    for tick in range(5):
        out=b._tick({ch:2. for ch in PATTERNS[label]} if tick==0 else {})
        traces.append(b.eligibility[-1])
        if tick>=1:
            scores+=np.array([out[list(g)].sum() for g in MOTOR])
    tie=bool(scores[0]==scores[1])
    action=int(np.random.default_rng(noise^0xCAFE).integers(2)) if tie else int(scores.argmax())
    return action,tie,traces

def evaluate(b,seed):
    correct=ties=0
    cm=np.zeros((2,2),int)
    for k in range(16):
        for label in (0,1):
            action,tie,_=sample(b,label,seed*100000+50000+k)
            correct+=action==label
            ties+=tie
            cm[label,action]+=1
    return dict(correct=int(correct),n=32,ties=int(ties),confusion=cm.tolist())

def run(seed,output):
    rows=[]
    for mode in ("global","traced","frozen"):
        reference=None
        for reward_mode in (("contingent",) if mode=="frozen" else ("contingent","permuted")):
            b=HQ200K(seed,plasticity=True,excitation_gain=4.,inhibition_gain=8.,
                     adaptation_strength=1.,electrode_patches=1)
            before=evaluate(b,seed)
            labels=np.tile([0,1],32)
            np.random.default_rng(seed^12345).shuffle(labels)
            replay=None if reward_mode=="contingent" else np.random.default_rng(seed^0xBEEF).permutation(reference)
            rewards=[];l1=0.;eligible=[];correct=0;ties=0
            for k,label in enumerate(labels):
                action,tie,traces=sample(b,int(label),seed*100000+k)
                actual=1 if action==label else -1
                reward=actual if replay is None else int(replay[k])
                rewards.append(reward);correct+=actual>0;ties+=tie
                terminal=b.electrodes[list(MOTOR[action])].ravel()
                ids,strength=select_edges(b.dst,traces,terminal,mode)
                eligible.append(len(ids))
                if mode!="frozen":
                    l1+=update(b,ids,strength,reward)
            if replay is None:
                reference=rewards
            after=evaluate(b,seed)
            if mode=="frozen" and before!=after:
                raise RuntimeError("frozen control changed")
            rows.append(dict(seed=seed,mode=mode,reward_mode=reward_mode,before=before,after=after,
                             training_correct=int(correct),training_ties=int(ties),training_n=64,
                             reward_signs=rewards,mean_eligible=float(np.mean(eligible)),
                             empty_eligible=sum(x==0 for x in eligible),weight_l1=l1))
            del b
    result=dict(scope="fixed early motor mapping, action-index eligibility signal, no game or PPO; R18 dynamics not stability-qualified",rows=rows)
    output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print("HQ200K_ACTION_CREDIT_RESULT",json.dumps(result,allow_nan=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.seed,a.output)
