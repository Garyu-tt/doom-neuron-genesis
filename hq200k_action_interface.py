"""R24: local deterministic motor readout and synchronous action credit.
No PPO or target-action labels. Designed for a separate game experiment.
"""
import numpy as np
from genesis_neural_interface import GROUPS
from benchmarks.hq200k_action_credit import select_edges,update

def motor_action(counts):
    """Eight group counts: encoding ignored; ties yield idle; no speed group."""
    x=np.asarray(counts)
    if x.shape!=(8,) or not np.isfinite(x).all() or (x<0).any():
        raise ValueError("expected eight finite nonnegative counts")
    def axis(a,b):
        return 1 if a>b else 2 if b>a else 0
    return (axis(x[1],x[2]),axis(x[3],x[4]),axis(x[5],x[6]),int(x[7]>0),0)

def action_channels(action):
    f,s,t,a,v=action
    if f not in (0,1,2) or s not in (0,1,2) or t not in (0,1,2) or a not in (0,1) or v!=0:
        raise ValueError("unsupported fixed motor action")
    groups=[]
    for choice,offset in ((f,0),(s,2),(t,4)):
        if choice:
            groups.extend(GROUPS[offset+choice][1])
    if a:
        groups.extend(GROUPS[7][1])
    return tuple(groups)

class ActionCredit:
    """Exactly one credit application per decision. Idle or neutral reward skips."""
    def __init__(self,brain,mode="traced",budget=.1):
        if mode not in ("traced","global","frozen") or not 0<budget<=.1:
            raise ValueError("invalid credit mode or budget")
        self.brain=brain
        self.mode=mode
        self.budget=budget
        self.pending=None
        self.decisions=self.updates=self.empty=self.neutral=0
        self.l1=0.
        self.cap_total=0.

    def decide(self,counts,traces):
        if self.pending is not None:
            raise RuntimeError("previous decision has no reward")
        action=motor_action(counts)
        # Snapshot the observed spike chains before subsequent ticks.
        self.pending=(action,[(r.copy(),s.copy()) for r,s in traces])
        self.decisions+=1
        return action

    def reward(self,scalar):
        if self.pending is None:
            raise RuntimeError("reward without decision")
        if not np.isfinite(scalar):
            raise ValueError("nonfinite reward")
        action,traces=self.pending
        self.pending=None
        if scalar==0:
            self.neutral+=1
            return
        channels=action_channels(action)
        if not channels:
            self.empty+=1
            return
        b=self.brain
        terminal=b.electrodes[list(channels)].ravel()
        ids,strength=select_edges(b.dst,traces,terminal,"traced")
        cap=min(self.budget,float(strength.sum()))
        self.cap_total+=cap
        if not cap:
            self.empty+=1
            return
        if self.mode=="frozen":
            return
        if self.mode=="global":
            ids,strength=select_edges(b.dst,traces,terminal,"global")
        delta=update(b,ids,strength,1 if scalar>0 else -1,budget=cap)
        self.l1+=delta
        self.updates+=int(delta>0)

    def telemetry(self):
        return dict(mode=self.mode,decisions=self.decisions,updates=self.updates,
                    empty=self.empty,neutral=self.neutral,l1=self.l1,cap_total=self.cap_total)
