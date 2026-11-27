"""Paired causal input/readout calibration, no gameplay or reward optimization.
Reuse the exact HQ200K topology and neuron update. Gain and pulse variants are
diagnostic perturbations, not changes to the reference backend.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from genesis_neural_interface import GROUPS
from hq200k_backend import HQ200K, ENCODING, N

MOTOR_CHANNELS = sorted({ch for _, channels in GROUPS[1:] for ch in channels})
VARIANTS = (
    ("reference", 1., "single"),
    ("gain2", 2., "single"),
    ("gain4", 4., "single"),
    ("train_40hz", 1., "train"),
    ("gain2_train_40hz", 2., "train"),
)

def reset(b, noise_seed):
    b.v.fill(0)
    b.current.fill(0)
    b.refractory.fill(0)
    b.previous = np.empty(0,np.int32)
    b.feedback_queue.clear()
    b.eligibility.clear()
    b.rng = np.random.default_rng(noise_seed)
    b.ticks = b.spikes = b.steps = b.updates = b.feedback_events = 0

def trace(b, noise_seed, pulse_mode, enabled):
    reset(b, noise_seed)
    # 100 ms preconditioning, then 100 ms experiment; common random numbers.
    for _ in range(10):
        b._tick({})
    counts=[]
    activity=[]
    for t in range(10):
        # 40 Hz rounded to the 10 ms time grid, explicitly a diagnostic train.
        pulse = t==0 if pulse_mode=="single" else t in (0,3,5,8)
        counts.append(b._tick({ch:2.0 for ch in ENCODING} if enabled and pulse else {}))
        activity.append(b.previous.copy())
    return np.asarray(counts), activity

def paired(b, noise_seed, mode):
    control, csp = trace(b,noise_seed,mode,False)
    stim, ssp = trace(b,noise_seed,mode,True)
    delta=stim[:,MOTOR_CHANNELS]-control[:,MOTOR_CHANNELS]
    return {
        "motor_l1_0_100ms": int(np.abs(delta).sum()),
        "motor_l1_50_100ms": int(np.abs(delta[5:]).sum()),
        "motor_l1_integrated_50_100ms": int(np.abs(delta[5:].sum(axis=0)).sum()),
        "motor_spikes_stim_50_100ms": int(stim[5:,MOTOR_CHANNELS].sum()),
        "motor_spikes_control_50_100ms": int(control[5:,MOTOR_CHANNELS].sum()),
        "population_symdiff_per_tick":[int(len(np.setxor1d(a,c,assume_unique=True))) for a,c in zip(ssp,csp)],
        "stim_total_spikes":int(sum(map(len,ssp))),
        "control_total_spikes":int(sum(map(len,csp))),
    }

def main(output):
    rows=[]
    alias=[]
    for seed in (37421,37422,37423):
        b=HQ200K(seed,plasticity=False)
        original=b.w.copy()
        # Exact backend test: does the upstream 8-vector frequency survive?
        reset(b,seed+1000)
        a=b.stimulate([4.]*8,[1.]*8,GROUPS,50,50)
        reset(b,seed+1000)
        c=b.stimulate([40.]*8,[2.5]*8,GROUPS,50,50)
        alias.append({"seed":seed,"low_readout":a,"high_readout":c,"equal":a==c})
        for name,gain,mode in VARIANTS:
            b.w[:]=original*gain
            for k in range(4):
                rows.append({"variant":name,"graph_seed":seed,"noise_seed":seed*100+k,
                             **paired(b,seed*100+k,mode)})
        del original,b
    summary={}
    for name,_,_ in VARIANTS:
        r=[x for x in rows if x["variant"]==name]
        summary[name]={"pairs":len(r),
                       "pairs_motor_effect_after_50ms":sum(x["motor_l1_50_100ms"]>0 for x in r),
                       "mean_motor_l1_after_50ms":float(np.mean([x["motor_l1_50_100ms"] for x in r])),
                       "mean_control_spikes":float(np.mean([x["control_total_spikes"] for x in r])),
                       "mean_stim_spikes":float(np.mean([x["stim_total_spikes"] for x in r]))}
    data={"scope":"causal calibration only; same graph and noise within pairs; no learning",
          "alias_test":alias,"summary":summary,"paired_trials":rows}
    output.write_text(json.dumps(data,indent=2,allow_nan=False))
    print("HQ200K_CAUSAL_RESULT",json.dumps(data,allow_nan=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args().output)
