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


class HQ200K:
    def __init__(self, seed=37421, plasticity=True, shuffled=False):
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
                "tick_ms":TICK_MS,"electrode_cells":self.electrodes.shape[1]}

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
        firing = np.flatnonzero(ready & (self.v >= 1.)).astype(np.int32)
        self.v[firing] = 0
        self.refractory[firing] = 2
        self.current.fill(0)
        if len(firing):
            np.add.at(self.current, self.dst[firing].ravel(), self.w[firing].ravel())
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
        # Upstream burst_count=1; one pulse per positive-frequency encoding channel.
        for t in range(wait+collect):
            injection = {ch:a for ch,f,a in zip(ENCODING,frequencies_hz,amplitudes_ua)
                         if t==0 and f>0 and a!=0}
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
            for age,(rows,slots) in enumerate(reversed(self.eligibility)):
                if len(rows):
                    old = self.w[rows,slots]
                    self.w[rows,slots] = np.sign(old)*np.clip(np.abs(old)+sign*0.001*(0.8**age),0.001,0.25)
                    self.updates += len(rows)

    def event(self, kind, data):
        # State persists across game episodes; metadata is not an input to the model.
        if kind == "episode_end":
            print(json.dumps({"hq200k":self.telemetry()},sort_keys=True),flush=True)

    def telemetry(self):
        return {**self.metadata(),"steps":self.steps,"ticks":self.ticks,"spikes":self.spikes,
                "plasticity_updates":self.updates,"feedback_events":self.feedback_events}

    def close(self):
        print(json.dumps({"hq200k":self.telemetry()},sort_keys=True),flush=True)


def factory(topology_path, neurons, seed):
    """Config is explicit; graph is reproducibly materialized and hashed at startup."""
    if neurons!=N:
        raise ValueError("HQ200K requires 200000 resident neurons")
    cfg = json.loads(Path(topology_path).read_text())
    if not isinstance(cfg,dict) or set(cfg)-{"plasticity","shuffled","expected_sha256"}:
        raise ValueError("unknown HQ200K config field")
    for key in ("plasticity","shuffled"):
        if key in cfg and type(cfg[key]) is not bool:
            raise ValueError("expected boolean config")
    brain = HQ200K(seed,cfg.get("plasticity",True),cfg.get("shuffled",False))
    if cfg.get("expected_sha256") and cfg["expected_sha256"]!=brain.initial_hash:
        raise ValueError("topology hash mismatch")
    return brain
