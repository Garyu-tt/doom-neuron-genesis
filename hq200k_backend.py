# SPDX-License-Identifier: GPL-3.0-or-later
"""HQ500M-derived compact CPU spiking model, modified 2026-11-27.
Source: Garyu-tt/mikoshi-hq500m, hq500m/core.py, c99d191ef5aab22465d32fd3e05f76447c3debf5.
Synthetic topology and virtual electrodes; no anatomical or learning claims.
"""
import hashlib
import json
import math
from collections import deque
from pathlib import Path
import numpy as np

N = 200_000
SIZES = (90_000, 28_000, 20_000, 20_000, 16_000, 14_000, 12_000)
NAMES = ("cortex", "cerebellum", "hippocampus", "thalamus",
         "basal_ganglia", "brainstem", "limbic")
ROUTES = (3, 5, 0, 0, 3, 1, 2)
BOUNDS = np.array((0, *np.cumsum(SIZES)), dtype=np.int64)
MODULE = 1024
K = 0x517cc1b727220a95
TICK_MS = 10
ENCODING = (8, 9, 10, 17, 18, 25, 27, 28)
POS, NEG = {19, 20, 22}, {23, 24, 26}


def mix(x):
    """Vectorized unsigned SplitMix64; overflow is intentional."""
    x = x + np.uint64(0x9e3779b97f4a7c15)
    x = (x ^ (x >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> 27)) * np.uint64(0x94d049bb133111eb)
    return x ^ (x >> 31)


def build_graph(seed=37421, shuffled=False):
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("invalid seed")
    dst = np.empty((N, 32), dtype=np.int32)
    w = np.empty((N, 32), dtype=np.float32)
    for r, (lo, hi) in enumerate(zip(BOUNDS[:-1], BOUNDS[1:])):
        ids = np.arange(lo, hi, dtype=np.uint64)
        h0 = mix(ids ^ np.uint64(seed))
        inhibitory = (h0 % 100) < 20
        local = ids - np.uint64(lo)
        base = local // MODULE * MODULE
        count = np.minimum(MODULE, np.uint64(hi-lo)-base)
        odd = ((h0 >> 11) & np.uint64(0xffffffff)) | np.uint64(1)
        target = ROUTES[r]
        tlo, thi = map(int, BOUNDS[target:target+2])
        for slot in range(32):
            h = mix(h0 ^ np.uint64(((slot+1)*K) % 2**64))
            if slot < 24:
                to = np.uint64(lo)+base+(((local-base+(slot+1)*odd)&np.uint64(0xffffffff))%count)
            else:
                blocks = (thi-tlo+MODULE-1)//MODULE
                tbase = mix(h ^ (local//MODULE)) % blocks * MODULE
                tcount = np.minimum(MODULE, np.uint64(thi-tlo)-tbase)
                to = np.uint64(tlo)+tbase+(h>>24)%tcount
            dst[lo:hi,slot] = to
            w[lo:hi,slot] = (300+((h>>33)%4800)).astype(np.float32)*np.float32(0.000035)*np.where(inhibitory,-1,1)
    if shuffled:
        # Permute target slots globally: preserves exact in/out degree, not regional routing.
        np.random.default_rng(seed ^ 0xD00).shuffle(dst.ravel())
    return dst, w


def graph_hash(dst, w):
    h = hashlib.sha256()
    h.update(dst.astype("<i4", copy=False).tobytes())
    h.update(w.astype("<f4", copy=False).tobytes())
    return h.hexdigest()


def patch_electrodes(seed, patches):
    """Exact R17/R18 synthetic placement; no graph-edge inspection."""
    if type(patches) is not int or patches not in (1,2,4):
        raise ValueError("expected one, two or four modules")
    rng=np.random.default_rng(seed ^ 0xACED)
    parts=[]
    for i,(lo,hi) in enumerate(zip(BOUNDS[:-1],BOUNDS[1:])):
        width=5 if i<4 else 4
        if patches==1:
            base=int(lo)+MODULE*int(rng.integers(int(hi-lo)//MODULE))
            parts.append((base+rng.choice(MODULE,64*width,replace=False)).reshape(64,width))
            continue
        bases=int(lo)+MODULE*rng.choice(int(hi-lo)//MODULE,patches,replace=False)
        chunks=[]
        for p,base in enumerate(bases):
            count=width//patches+int(p<width%patches)
            if count:
                chunks.append((base+rng.choice(MODULE,64*count,replace=False)).reshape(64,count))
        parts.append(np.concatenate(chunks,axis=1))
    return np.concatenate(parts,axis=1)


class HQ200K:
    def __init__(self, seed=37421, plasticity=True, shuffled=False, excitation_gain=1., inhibition_gain=1., stimulus_mode="single", adaptation_strength=0., adaptation_tau_ms=200., plasticity_mode="legacy", plasticity_budget=2., electrode_patches=0):
        for gain in (excitation_gain, inhibition_gain):
            if isinstance(gain, bool) or not isinstance(gain,(int,float)) or not math.isfinite(gain) or not 0 < gain <= 8:
                raise ValueError('gain must be finite, positive and <=8')
        if stimulus_mode not in ('single', 'rate'):
            raise ValueError('stimulus_mode must be single or rate')
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in (adaptation_strength,adaptation_tau_ms)) or not 0 <= adaptation_strength <= 4 or not 20 <= adaptation_tau_ms <= 2000:
            raise ValueError("invalid adaptation parameters")
        if plasticity_mode not in ("legacy","budgeted") or isinstance(plasticity_budget,bool) or not isinstance(plasticity_budget,(int,float)) or not math.isfinite(plasticity_budget) or not 0 < plasticity_budget <= 100:
            raise ValueError("invalid plasticity budget")
        self.plasticity_mode = plasticity_mode
        self.plasticity_budget = float(plasticity_budget)
        self.plasticity_l1 = 0.
        self.plasticity_events = 0
        self.adaptation_strength = float(adaptation_strength)
        self.adaptation_tau_ms = float(adaptation_tau_ms)
        self.adaptation_decay = math.exp(-TICK_MS/self.adaptation_tau_ms)
        self.adaptation = np.zeros(N,np.float32)
        self.stimulus_mode = stimulus_mode
        self.stim_phase = np.zeros(8, np.float64)
        self.input_pulses = np.zeros(8, np.int64)
        self.excitation_gain = float(excitation_gain)
        self.inhibition_gain = float(inhibition_gain)
        self.dst, self.w = build_graph(seed, shuffled)
        self.initial_hash = graph_hash(self.dst, self.w)
        self.seed, self.shuffled, self.plasticity = seed, shuffled, plasticity
        self.rng = np.random.default_rng(seed ^ 0xA11CE)
        self.v = np.zeros(N, np.float32)
        self.current = np.zeros(N, np.float32)
        self.refractory = np.zeros(N, np.uint8)
        # Fixed abstract electrodes: 32 cells per channel, no direct motor shortcuts.
        # Each channel samples all seven regions; no two channels share a neuron.
        chosen = [self.rng.choice(int(hi-lo), 64*5, replace=False)+lo
                  for lo, hi in zip(BOUNDS[:-1], BOUNDS[1:])]
        self.electrodes = np.concatenate([x.reshape(64,5)[:, :5 if i<4 else 4]
                                         for i,x in enumerate(chosen)],axis=1)
        if type(electrode_patches) is not int or electrode_patches not in (0,1,2,4):
            raise ValueError("electrode_patches must be 0, 1, 2 or 4")
        self.electrode_patches=electrode_patches
        if electrode_patches:
            self.electrodes=patch_electrodes(seed,electrode_patches)
        self.electrode_sha256=hashlib.sha256(self.electrodes.astype("<i8",copy=False).tobytes()).hexdigest()
        self.previous = np.empty(0, np.int32)
        self.eligibility = deque(maxlen=10)
        self.feedback_queue = {}
        self.ticks = self.spikes = self.updates = 0
        self.steps = self.feedback_events = 0

    def metadata(self):
        return {"architecture":"HQ200K","regions":len(SIZES),
                "region_sizes":dict(zip(NAMES,SIZES)), "resident_neurons":len(self.v),
                "synapse_slots":int(self.dst.size),"topology_sha256":self.initial_hash,
                "seed":self.seed,"shuffled":self.shuffled,"plasticity":self.plasticity,
                "tick_ms":TICK_MS,"electrode_cells":self.electrodes.shape[1],
                "electrode_patches":self.electrode_patches,"electrode_sha256":self.electrode_sha256,
                "excitation_gain":self.excitation_gain,"inhibition_gain":self.inhibition_gain,"stimulus_mode":self.stimulus_mode,
                "adaptation_strength":self.adaptation_strength,"adaptation_tau_ms":self.adaptation_tau_ms,
                "plasticity_mode":self.plasticity_mode,"plasticity_budget":self.plasticity_budget}

    def _tick(self, injection):
        # Original HQ500M leak, threshold, refractory and weight scale.
        # A declared stochastic spontaneous-drive assumption prevents a dead start.
        noise = self.rng.choice(N, N//1000, replace=False)
        self.current[noise] += 1.05
        for ch, amp in injection.items():
            self.current[self.electrodes[ch]] += min(4., abs(amp)*1.1)
        ready = self.refractory == 0
        self.v[ready] = np.clip(self.v[ready]*0.68+self.current[ready], -2., None)
        self.v[~ready] = 0
        self.refractory[~ready] -= 1
        # Local spike-triggered threshold adaptation, no reward or game input.
        if self.adaptation_strength:
            self.adaptation *= self.adaptation_decay
            firing = np.flatnonzero(ready & (self.v >= 1.+self.adaptation)).astype(np.int32)
            self.adaptation[firing] = np.minimum(8.,self.adaptation[firing]+self.adaptation_strength)
        else:
            firing = np.flatnonzero(ready & (self.v >= 1.)).astype(np.int32)
        self.v[firing] = 0
        self.refractory[firing] = 2
        self.current.fill(0)
        if len(firing):
            weights = self.w[firing]
            if self.excitation_gain != 1. or self.inhibition_gain != 1.:
                weights = weights * np.where(weights > 0, self.excitation_gain, self.inhibition_gain).astype(np.float32)
            np.add.at(self.current, self.dst[firing].ravel(), weights.ravel())
        if self.plasticity:
            # Causal pre(t-1) -> post(t) eligibility, limited to last 100 virtual ms.
            mask = np.zeros(N, bool)
            mask[firing] = True
            rr, ss = np.nonzero(mask[self.dst[self.previous]])
            self.eligibility.append((self.previous[rr], ss))
        self.previous = firing
        self.ticks += 1
        self.spikes += len(firing)
        hit = np.zeros(N, bool)
        hit[firing] = True
        return hit[self.electrodes].sum(axis=1)

    def stimulate(self, frequencies_hz, amplitudes_ua, groups, artifact_wait_ms, collect_window_ms):
        if len(frequencies_hz)!=8 or len(amplitudes_ua)!=8 or len(groups)!=8:
            raise ValueError("expected eight inputs and readout groups")
        if tuple(groups[0][1]) != ENCODING:
            raise ValueError("encoding channel mismatch")
        if any(not math.isfinite(x) for x in (*frequencies_hz,*amplitudes_ua,artifact_wait_ms,collect_window_ms)):
            raise ValueError("nonfinite input")
        if not 0<=artifact_wait_ms<=1000 or not 0<collect_window_ms<=1000:
            raise ValueError("bounded virtual time required")
        wait = math.ceil(artifact_wait_ms/TICK_MS)
        collect = math.ceil(collect_window_ms/TICK_MS)
        counts = np.zeros(64, np.int64)
        # Reference uses the upstream single pulse. Experimental rate mode
        # preserves fractional phase across packets; 0..40 Hz is the upstream range.
        freq = np.asarray(frequencies_hz, np.float64)
        amps = np.asarray(amplitudes_ua, np.float64)
        if np.any(freq < 0) or (self.stimulus_mode == "rate" and np.any(freq > 100)):
            raise ValueError("unsupported frequency")
        for t in range(wait+collect):
            if self.stimulus_mode == "rate":
                self.stim_phase += freq * (TICK_MS/1000)
                pulses = np.floor(self.stim_phase + 1e-12).astype(np.int64)
                self.stim_phase -= pulses
            else:
                pulses = ((freq>0) & (t==0)).astype(np.int64)
            pulses = np.where(amps!=0,pulses,0)
            self.input_pulses += pulses
            injection = {ch:float(a*p) for ch,a,p in zip(ENCODING,amps,pulses) if p}
            for ch,(left,hz,amp,phase) in list(self.feedback_queue.items()):
                phase += hz*TICK_MS/1000
                pulses = min(left,int(phase))
                if pulses:
                    injection[ch] = injection.get(ch,0)+amp*pulses
                if left==pulses:
                    del self.feedback_queue[ch]
                else:
                    self.feedback_queue[ch] = (left-pulses,hz,amp,phase-int(phase))
            observed = self._tick(injection)
            if t>=wait:
                counts += observed
        self.steps += 1
        return [int(counts[list(channels)].sum()) for _,channels in groups]

    def feedback(self, kind, channels, frequency_hz, amplitude_ua, pulses, unpredictable, event_name):
        channels = set(channels)
        if not channels <= set(range(64)):
            raise ValueError("invalid channels")
        if kind == "interrupt":
            for ch in channels:
                self.feedback_queue.pop(ch,None)
            return
        if kind not in ("reward","event") or not channels or frequency_hz<=0 or amplitude_ua<=0 or pulses<=0:
            return
        self.feedback_events += 1
        for ch in channels:
            self.feedback_queue[ch] = (min(int(pulses),1000),min(float(frequency_hz),500.),min(float(amplitude_ua),4.),0.)
        # Experimental reward-modulated local synaptic rule; no external action scores.
        sign = 1 if kind=="reward" and channels<=POS else -1 if kind=="reward" and channels<=NEG else 0
        if self.plasticity and sign:
            self.plasticity_events += 1
            if self.plasticity_mode == "budgeted":
                # Coalesce repeated eligibility using its strongest recent trace.
                ids=[]; strength=[]
                for age,(rows,slots) in enumerate(reversed(self.eligibility)):
                    if len(rows):
                        ids.append(rows.astype(np.int64)*32+slots)
                        strength.append(np.full(len(rows),0.001*(0.8**age),np.float64))
                if ids:
                    unique,inverse=np.unique(np.concatenate(ids),return_inverse=True)
                    delta=np.zeros(len(unique),np.float64)
                    np.maximum.at(delta,inverse,np.concatenate(strength))
                    # Keep total update magnitude bounded independent of burst size.
                    delta *= min(1.,self.plasticity_budget/max(float(delta.sum()),1e-12))
                    rows,slots=unique//32,unique%32
                    old=self.w[rows,slots]
                    new=(np.sign(old)*np.clip(np.abs(old)+sign*delta,0.001,0.25)).astype(np.float32)
                    self.w[rows,slots]=new
                    self.plasticity_l1 += float(np.abs(new-old).sum(dtype=np.float64))
                    self.updates += int(np.count_nonzero(new!=old))
            else:
                for age,(rows,slots) in enumerate(reversed(self.eligibility)):
                    if len(rows):
                        old = self.w[rows,slots]
                        new = np.sign(old)*np.clip(np.abs(old)+sign*0.001*(0.8**age),0.001,0.25)
                        self.w[rows,slots] = new
                        self.plasticity_l1 += float(np.abs(self.w[rows,slots]-old).sum(dtype=np.float64))
                        self.updates += len(rows)

    def event(self, kind, data):
        # State persists across game episodes; metadata is not an input to the model.
        if kind == "episode_end":
            print(json.dumps({"hq200k":self.telemetry()},sort_keys=True),flush=True)

    def telemetry(self):
        return {**self.metadata(),"steps":self.steps,"ticks":self.ticks,"spikes":self.spikes,
                "plasticity_updates":self.updates,"feedback_events":self.feedback_events,
                "input_pulses":self.input_pulses.tolist(),"plasticity_l1":self.plasticity_l1,
                "plasticity_events":self.plasticity_events}

    def close(self):
        print(json.dumps({"hq200k":self.telemetry()},sort_keys=True),flush=True)


def factory(topology_path, neurons, seed):
    """Config is explicit; graph is reproducibly materialized and hashed at startup."""
    if neurons!=N:
        raise ValueError("HQ200K requires 200000 resident neurons")
    cfg = json.loads(Path(topology_path).read_text())
    if not isinstance(cfg,dict) or set(cfg)-{"plasticity","shuffled","expected_sha256","excitation_gain","inhibition_gain","stimulus_mode","adaptation_strength","adaptation_tau_ms","plasticity_mode","plasticity_budget","electrode_patches"}:
        raise ValueError("unknown HQ200K config field")
    for key in ("plasticity","shuffled"):
        if key in cfg and type(cfg[key]) is not bool:
            raise ValueError("expected boolean config")
    brain = HQ200K(seed,cfg.get("plasticity",True),cfg.get("shuffled",False),
                   cfg.get("excitation_gain",1.),cfg.get("inhibition_gain",1.),cfg.get("stimulus_mode","single"),
                   cfg.get("adaptation_strength",0.),cfg.get("adaptation_tau_ms",200.),
                   cfg.get("plasticity_mode","legacy"),cfg.get("plasticity_budget",2.),cfg.get("electrode_patches",0))
    if cfg.get("expected_sha256") and cfg["expected_sha256"]!=brain.initial_hash:
        raise ValueError("topology hash mismatch")
    return brain
