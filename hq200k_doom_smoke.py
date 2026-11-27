"""Bounded real ViZDoom/CL1-UDP integration experiment using upstream PPOTrainer.
Shortened episode timeout is for smoke testing, not a biological benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import resource
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch
import vizdoom
import udp_protocol as wire
from genesis_neural_interface import serve
from hq200k_backend import HQ200K
from training_server import PPOConfig, PPOTrainer


def run(arm, seed, output, batches):
    output.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    source = Path("progressive_deathmatch.cfg").resolve()
    wad = source.with_name("progressive_deathmatch.wad")
    cfg = output / "smoke.cfg"
    text = source.read_text()
    text = re.sub(r"(?m)^doom_scenario_path\s*=.*$",f"doom_scenario_path = {os.path.relpath(wad, cfg.parent.resolve())}",text)
    text = re.sub(r"(?m)^episode_timeout\s*=.*$","episode_timeout = 128",text)
    cfg.write_text(text)
    brain = HQ200K(seed, plasticity=arm!="frozen", shuffled=arm=="shuffled")
    args = SimpleNamespace(stim_port=12345,spike_port=12346,event_port=12347,
                           feedback_port=12348,max_steps=1000,idle_timeout=30,
                           artifact_wait_ms=50,collect_window_ms=50)
    config = PPOConfig(doom_config=str(cfg),steps_per_update=32,batch_size=32,num_epochs=1,
                       max_episodes=10,checkpoint_dir=str(output/"checkpoints"),
                       log_dir=str(output/"tensorboard"),vis_host="127.0.0.1",
                       episode_reset_delay_s=0.0, decoder_ablation_mode="none")
    trainer = PPOTrainer(config, tick_frequency_hz=10, show_window=False, device="cpu")
    trainer.env.game.set_seed(seed)
    trainer.setup_udp_sockets()
    errors=[]
    def worker():
        try:
            serve(brain,args)
        except Exception as exc:
            errors.append(repr(exc))
    thread=threading.Thread(target=worker,daemon=True)
    thread.start()
    # Allow bounded local bind; a strict receive below rejects missing packets.
    time.sleep(.1)
    received=[]
    rng=np.random.default_rng(seed^0x123)
    def strict_receive(sock):
        time.sleep(config.spike_artifact_wait_s)
        pkt,addr=sock.recvfrom(65535)
        if addr[0]!="127.0.0.1":
            raise RuntimeError("nonlocal spike packet")
        _, counts=wire.unpack_spike_data(pkt)
        if not np.isfinite(counts).all() or (counts<0).any():
            raise RuntimeError("invalid spike data")
        received.append(counts.tolist())
        if arm=="zero":
            return np.zeros(8,np.float32)
        if arm=="random":
            return rng.poisson(1.0,size=8).astype(np.float32)
        return counts
    trainer.policy.collect_spikes=strict_receive
    progress=[]
    start=time.perf_counter()
    cpu=time.process_time()
    try:
        for i in range(batches):
            prior=trainer.total_steps
            rx=len(received)
            rollout=trainer._collect_rollouts_hardware(32,10)
            actual=trainer.total_steps-prior
            if actual<32 or len(received)-rx!=actual or errors:
                raise RuntimeError(f"incomplete rollout: steps={actual}, packets={len(received)-rx}, errors={errors}")
            features=rollout["spike_features"].detach().cpu().numpy()
            trainer.update_policy(rollout)
            progress.append({"batch":i+1,"steps":trainer.total_steps,"episodes":trainer.total_episodes,
                             "episode_rewards":list(trainer.episode_rewards),
                             "episode_lengths":list(trainer.episode_lengths),
                             "killcounts":list(trainer.episode_killcounts),
                             "decoder_nonzero_fraction":float(np.count_nonzero(features)/features.size),
                             "decoder_spike_mean":features.mean(axis=0).tolist()})
        trainer.event_socket.sendto(wire.pack_event_metadata("training_complete",{}),("127.0.0.1",12347))
        thread.join(5)
        if thread.is_alive() or errors or brain.steps!=len(received):
            raise RuntimeError(f"bridge did not complete cleanly: {errors}, {brain.steps}/{len(received)}")
        result={"status":"smoke_only","arm":arm,"seed":seed,
                "scenario_sha256":hashlib.sha256(wad.read_bytes()).hexdigest(),
                "original_config_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                "episode_timeout_tics":128,"vizdoom":vizdoom.__version__,
                "numpy":np.__version__,"torch":torch.__version__,
                "upstream_ppo":True,"random_control_poisson_mean":1.0,
                "steps":trainer.total_steps,"packets":len(received),
                "raw_spike_mean":np.mean(received,axis=0).tolist(),
                "raw_nonzero_fraction":float(np.count_nonzero(received)/np.size(received)),
                "neural":brain.telemetry(),"progress":progress,
                "wall_s":time.perf_counter()-start,"cpu_s":time.process_time()-cpu,
                "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        (output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False))
        (output/"raw_spikes.json").write_text(json.dumps(received))
        print("HQ200K_GAME_RESULT",json.dumps(result,allow_nan=False),flush=True)
    finally:
        if thread.is_alive():
            trainer.event_socket.sendto(wire.pack_event_metadata("training_complete",{}),("127.0.0.1",12347))
            thread.join(2)
        for name in ("stim_socket","spike_socket","event_socket","feedback_socket"):
            sock=getattr(trainer,name,None)
            if sock: sock.close()
        trainer.writer.close()
        trainer.env.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=("structured","frozen","shuffled","zero","random"),default="structured")
    p.add_argument("--seed",type=int,default=37421)
    p.add_argument("--batches",type=int,choices=(1,2,3,4),default=2)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.arm,a.seed,a.output,a.batches)
