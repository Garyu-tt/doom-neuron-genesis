"""R23: independent replication with learning curves and anti-reward control.
Global and traced controls use the same rule and per-trial update ceiling:
the available traced eligibility mass, capped at 0.1. No correct label
enters edge selection; scalar reward alone evaluates the executed action.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from hq200k_backend import HQ200K
from benchmarks.hq200k_action_credit import MOTOR,sample,select_edges,update

def evaluate(b,seed,flip):
    correct=ties=0
    cm=np.zeros((2,2),int)
    for k in range(64):
        for label in (0,1):
            action,tie,_=sample(b,label,seed*100000+70000+k)
            target=label^flip
            correct+=action==target
            ties+=tie
            cm[target,action]+=1
    return dict(correct=int(correct),n=128,ties=int(ties),confusion=cm.tolist())

def run(seed,output):
    rows=[]
    for flip in (0,1):
        for mode in ("traced","frozen"):
            reference=None
            for reward_mode in (("contingent",) if mode=="frozen" else ("contingent","permuted","inverted")):
                b=HQ200K(seed,plasticity=True,excitation_gain=3.75,inhibition_gain=8.,
                         adaptation_strength=1.5,adaptation_tau_ms=200.,electrode_patches=1)
                before=evaluate(b,seed,flip)
                initial=hashlib.sha256(b.w.tobytes()).hexdigest()
                labels=np.tile([0,1],96)
                np.random.default_rng(seed^12345).shuffle(labels)
                replay=np.random.default_rng(seed^0xBEEF).permutation(reference) if reward_mode=="permuted" else None
                rewards=[];l1=0.;eligible=[];caps=[];correct=0;ties=0;curve=[]
                for k,label in enumerate(labels):
                    action,tie,traces=sample(b,int(label),seed*100000+k)
                    actual=1 if action==(int(label)^flip) else -1
                    reward=-actual if reward_mode=="inverted" else actual if replay is None else int(replay[k])
                    rewards.append(reward);correct+=actual>0;ties+=tie
                    terminal=b.electrodes[list(MOTOR[action])].ravel()
                    tid,ts=select_edges(b.dst,traces,terminal,"traced")
                    cap=min(.1,float(ts.sum()))
                    ids,strength=(tid,ts) if mode=="traced" else select_edges(b.dst,traces,terminal,"global")
                    eligible.append(len(ids));caps.append(cap)
                    if mode!="frozen" and cap>0:
                        l1+=update(b,ids,strength,reward,budget=cap)
                    if (k+1) in (48,96,192):
                        curve.append(dict(training_trials=k+1,assessment=evaluate(b,seed,flip)))
                if reward_mode=="contingent":
                    reference=rewards
                after=evaluate(b,seed,flip)
                final=hashlib.sha256(b.w.tobytes()).hexdigest()
                if mode=="frozen" and (before!=after or initial!=final):
                    raise RuntimeError("frozen paired control changed")
                if l1>sum(caps)+.01:
                    raise RuntimeError("update exceeded declared cumulative ceiling")
                rows.append(dict(seed=seed,mapping=flip,mode=mode,reward_mode=reward_mode,
                                 before=before,after=after,curve=curve,training_correct=int(correct),
                                 training_ties=int(ties),training_n=len(labels),reward_signs=rewards,
                                 mean_eligible=float(np.mean(eligible)),empty_eligible=sum(x==0 for x in eligible),
                                 zero_cap=sum(x==0 for x in caps),total_cap=sum(caps),weight_l1=l1,
                                 initial_weight_sha256=initial,final_weight_sha256=final))
                print("R23_GROUP",json.dumps(rows[-1]),flush=True)
                del b
    data=dict(scope="stable R20 configuration; six new graphs; two opposite mappings; repeated fixed evaluation, no tuning on evaluation; true, permuted and inverted feedback; no PPO or game",
              chance=.5,rows=rows)
    output.write_text(json.dumps(data,indent=2,allow_nan=False))
    print("HQ200K_CREDIT_REPLICATION_RESULT",json.dumps(data,allow_nan=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.seed,a.output)
