"""R15: verify paired fresh-state evaluation of trained vs initial weights."""
import argparse,json
from pathlib import Path
from statistics import mean

def load(path):
    return json.loads((path/"result.json").read_text())

def metrics(d):
    p=d["progress"][-1]
    if not p["episode_rewards"]:
        raise ValueError("no complete evaluation episodes")
    return dict(episodes=p["episodes"],reward=mean(p["episode_rewards"]),
                kills=sum(p["killcounts"]),steps=d["steps"])

def report(root):
    train=load(root/"train")
    assert train["fixed_interface"] and train["initial_policy_sha256"]==train["final_policy_sha256"]
    pairs=[]
    for seed in (59101,59102):
        before=load(root/f"before-{seed}")
        after=load(root/f"after-{seed}")
        for d,expected in ((before,train["initial_neural_weights_sha256"]),(after,train["final_neural_weights_sha256"])):
            assert d["evaluation_seed"]==seed and d["evaluation_feedback_disabled"]
            assert d["fixed_interface"] and d["neural"]["feedback_events"]==0
            assert d["neural"]["plasticity_updates"]==0
            assert d["initial_neural_weights_sha256"]==d["final_neural_weights_sha256"]==expected
            assert d["initial_policy_sha256"]==d["final_policy_sha256"]==train["initial_policy_sha256"]
            assert d["neural"]["topology_sha256"]==train["neural"]["topology_sha256"]
            assert d["scenario_sha256"]==train["scenario_sha256"]
        a,b=metrics(before),metrics(after)
        pairs.append(dict(evaluation_seed=seed,before=a,after=b,delta_reward=b["reward"]-a["reward"]))
    result=dict(scope="paired held-out random seeds, same scenario; not evidence of transfer to new maps",
                graph_seed=train["seed"],mode=train["neural"]["plasticity_mode"],
                training_arm=train["arm"],training=metrics(train),pairs=pairs,
                weights_changed=train["initial_neural_weights_sha256"]!=train["final_neural_weights_sha256"],
                mean_delta_reward=mean(p["delta_reward"] for p in pairs))
    (root/"retention.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print("HQ200K_RETENTION_RESULT",json.dumps(result,allow_nan=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    report(p.parse_args().root)
