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


def policy_hash(policy):
    h=hashlib.sha256()
    for name,tensor in sorted(policy.state_dict().items()):
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def run(arm, seed, output, batches, excitation_gain=1., inhibition_gain=1., stimulus_mode="single", adaptation_strength=0., episode_tics=128, rollout_steps=32, plasticity_mode="legacy", plasticity_budget=2., fixed_interface=False, weights_path=None, save_weights=False, evaluation_seed=None, electrode_patches=0):
    if not (1<=batches<=16 and 128<=episode_tics<=4200 and 32<=rollout_steps<=256):
        raise ValueError("bounded experiment parameters required")
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
    text = re.sub(r"(?m)^episode_timeout\s*=.*$",f"episode_timeout = {episode_tics}",text)
    cfg.write_text(text)
    brain = HQ200K(seed, plasticity=arm!="frozen", shuffled=arm=="shuffled",
                   excitation_gain=excitation_gain, inhibition_gain=inhibition_gain, stimulus_mode=stimulus_mode, adaptation_strength=adaptation_strength, plasticity_mode=plasticity_mode, plasticity_budget=plasticity_budget, electrode_patches=electrode_patches)
    if evaluation_seed is not None and (not fixed_interface or arm!="frozen"):
        raise ValueError("held-out evaluation must freeze all weights")
    if weights_path is not None:
        with np.load(weights_path,allow_pickle=False) as data:
            if str(data["topology_sha256"].item())!=brain.initial_hash:
                raise ValueError("checkpoint topology mismatch")
            if "electrode_sha256" not in data or str(data["electrode_sha256"].item())!=brain.electrode_sha256:
                raise ValueError("checkpoint electrode mapping mismatch")
            weights=data["weights"]
            if weights.shape!=brain.w.shape or weights.dtype!=brain.w.dtype or not np.isfinite(weights).all() or not np.array_equal(np.sign(weights),np.sign(brain.w)):
                raise ValueError("invalid checkpoint weights")
            brain.w[:]=weights
    args = SimpleNamespace(stim_port=12345,spike_port=12346,event_port=12347,
                           feedback_port=12348,max_steps=batches*(rollout_steps+episode_tics)+1,idle_timeout=30,
                           artifact_wait_ms=50,collect_window_ms=50)
    config = PPOConfig(doom_config=str(cfg),steps_per_update=rollout_steps,batch_size=32,num_epochs=1,
                       max_episodes=10,checkpoint_dir=str(output/"checkpoints"),
                       log_dir=str(output/"tensorboard"),vis_host="127.0.0.1",
                       episode_reset_delay_s=0.0, decoder_ablation_mode="none",
                       center_spike_features=not fixed_interface)
    trainer = PPOTrainer(config, tick_frequency_hz=10, show_window=False, device="cpu")
    if fixed_interface:
        trainer.policy.eval()
        trainer.policy.requires_grad_(False)
    initial_policy_hash=policy_hash(trainer.policy)
    initial_weight_hash=hashlib.sha256(brain.w.tobytes()).hexdigest()
    trainer.strict_hardware_collection = True
    if evaluation_seed is not None:
        trainer._track_action_feedback=lambda *args,**kwargs: None
        trainer._send_episode_feedback=lambda *args,**kwargs: None
        random.seed(evaluation_seed)
        np.random.seed(evaluation_seed)
        torch.manual_seed(evaluation_seed)
        brain.rng=np.random.default_rng(evaluation_seed ^ 0xA11CE)
    trainer.env.game.set_seed(seed if evaluation_seed is None else evaluation_seed)
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
            episodes_before=trainer.total_episodes
            prior=trainer.total_steps
            rx=len(received)
            rollout=trainer._collect_rollouts_hardware(rollout_steps,10)
            actual=trainer.total_steps-prior
            if actual<rollout_steps or len(received)-rx!=actual or errors:
                raise RuntimeError(f"incomplete rollout: steps={actual}, packets={len(received)-rx}, errors={errors}")
            if trainer.total_episodes <= episodes_before:
                raise RuntimeError("rollout ended without a completed episode")
            features=rollout["spike_features"].detach().cpu().numpy()
            if not fixed_interface:
                trainer.update_policy(rollout)
            elif policy_hash(trainer.policy)!=initial_policy_hash:
                raise RuntimeError("fixed interface state changed")
            progress.append({"neural_weight_sha256":hashlib.sha256(brain.w.tobytes()).hexdigest(),
                             "plasticity_l1":brain.plasticity_l1,"batch":i+1,"steps":trainer.total_steps,"episodes":trainer.total_episodes,
                             "episode_rewards":list(trainer.episode_rewards),
                             "episode_lengths":list(trainer.episode_lengths),
                             "killcounts":list(trainer.episode_killcounts),
                             "decoder_nonzero_fraction":float(np.count_nonzero(features)/features.size),
                             "decoder_spike_mean":features.mean(axis=0).tolist()})
        trainer.event_socket.sendto(wire.pack_event_metadata("training_complete",{}),("127.0.0.1",12347))
        thread.join(5)
        if thread.is_alive() or errors or brain.steps!=len(received):
            raise RuntimeError(f"bridge did not complete cleanly: {errors}, {brain.steps}/{len(received)}")
        final_weight_hash=hashlib.sha256(brain.w.tobytes()).hexdigest()
        if arm=="frozen" and final_weight_hash!=initial_weight_hash:
            raise RuntimeError("frozen neural weights changed")
        if save_weights:
            np.savez(output/"weights.npz",weights=brain.w,topology_sha256=brain.initial_hash,electrode_sha256=brain.electrode_sha256)
        result={"evaluation_seed":evaluation_seed,"evaluation_feedback_disabled":evaluation_seed is not None,"initial_neural_weights_sha256":initial_weight_hash,"final_neural_weights_sha256":final_weight_hash,"status":"smoke_only" if batches<=4 and episode_tics==128 else "bounded_training_pilot","arm":arm,"seed":seed,
                "scenario_sha256":hashlib.sha256(wad.read_bytes()).hexdigest(),
                "original_config_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                "episode_timeout_tics":episode_tics,"rollout_steps_min":rollout_steps,"vizdoom":vizdoom.__version__,
                "numpy":np.__version__,"torch":torch.__version__,
                "upstream_ppo":not fixed_interface,"fixed_interface":fixed_interface,
                "initial_policy_sha256":initial_policy_hash,"final_policy_sha256":policy_hash(trainer.policy),"random_control_poisson_mean":1.0,
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
    p.add_argument("--batches",type=int,choices=range(1,17),default=2)
    p.add_argument("--excitation-gain",type=float,default=1.)
    p.add_argument("--inhibition-gain",type=float,default=1.)
    p.add_argument("--stimulus-mode",choices=("single","rate"),default="single")
    p.add_argument("--adaptation-strength",type=float,default=0.)
    p.add_argument("--episode-tics",type=int,default=128)
    p.add_argument("--rollout-steps",type=int,default=32)
    p.add_argument("--plasticity-mode",choices=("legacy","budgeted"),default="legacy")
    p.add_argument("--plasticity-budget",type=float,default=2.)
    p.add_argument("--fixed-interface",action="store_true")
    p.add_argument("--weights-path",type=Path)
    p.add_argument("--save-weights",action="store_true")
    p.add_argument("--evaluation-seed",type=int)
    p.add_argument("--electrode-patches",type=int,choices=(0,1,2,4),default=0)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.arm,a.seed,a.output,a.batches,a.excitation_gain,a.inhibition_gain,a.stimulus_mode,a.adaptation_strength,a.episode_tics,a.rollout_steps,a.plasticity_mode,a.plasticity_budget,a.fixed_interface,a.weights_path,a.save_weights,a.evaluation_seed,a.electrode_patches)
