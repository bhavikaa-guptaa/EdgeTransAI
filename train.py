"""
EdgeTransAI — Training Script (numpy-only fallback, no PyTorch required)
=========================================================================
Uses a random policy with simulated reward tracking so you can verify
the environment, data pipeline, and logging all work before installing
a full ML stack.

Run:
    python train.py                   # 500 steps, instant
    python train.py --steps 2000
"""

import os, sys, time, argparse, yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.simulation.traffic_env import TrafficEnv

# Try importing real agent — fall back to random if torch missing
try:
    import torch
    from src.agents.ppo_agent import PPOAgent
    from src.agents.doa import CityCoordinator
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[Train] PyTorch not found — using random policy (install torch for real training).")


class RandomAgent:
    """Dummy stand-in when PyTorch is not available."""
    def __init__(self, n_agents, action_dim):
        self.n = n_agents
        self.a = action_dim

    def select_actions(self, obs):
        acts = np.random.randint(0, self.a, self.n)
        return acts, np.zeros(self.n), np.zeros(self.n)

    def buffer_add(self, *_): pass
    def update(self, _): return {"loss_policy": 0.0, "loss_value": 0.0}
    def save(self, path): print(f"[RandomAgent] (no checkpoint — install torch) → {path}")


def make_dirs(config):
    for key in ["checkpoint_dir", "figures_dir", "logs_dir"]:
        os.makedirs(config["paths"][key], exist_ok=True)


def train(config, total_steps=500):
    make_dirs(config)
    np.random.seed(config["training"]["seed"])
    env = TrafficEnv(config)

    if TORCH_OK:
        import torch
        torch.manual_seed(config["training"]["seed"])
        agent = PPOAgent(
            state_dim=config["agent"]["state_dim"],
            action_dim=config["agent"]["action_dim"],
            config=config, device="cpu"
        )
        def add(o,a,lp,r,v,d): agent.buffer.add(o,a,lp,r,v,d)
        def upd(o): return agent.update(o)
        def save(p): agent.save(p)
    else:
        agent = RandomAgent(env.num_agents, config["agent"]["action_dim"])
        def add(*_): pass
        def upd(_): return {"loss_policy": 0.0}
        def save(p): agent.save(p)

    batch   = config["agent"]["batch_size"]
    log_ev  = max(50, total_steps // 15)
    save_ev = max(200, total_steps // 3)

    log = {"episode_reward": [], "mean_queue": []}
    obs = env.reset()
    ep_r = np.zeros(env.num_agents)
    episodes = 0
    t0 = time.time()

    print(f"[Train] Steps={total_steps}  Agents={env.num_agents}  "
          f"{'PPO' if TORCH_OK else 'Random'} policy")
    print("-" * 55)

    for step in range(1, total_steps + 1):
        actions, lp, vals = agent.select_actions(obs)
        next_obs, rewards, done, info = env.step(actions)
        add(obs, actions, lp, rewards, vals, done)
        ep_r += rewards
        obs = next_obs

        if done:
            log["episode_reward"].append(float(ep_r.mean()))
            log["mean_queue"].append(info["mean_queue"])
            ep_r[:] = 0; episodes += 1; obs = env.reset()

        if step % batch == 0:
            upd(obs)

        if step % log_ev == 0:
            r   = np.mean(log["episode_reward"][-5:]) if log["episode_reward"] else 0
            q   = np.mean(log["mean_queue"][-5:])     if log["mean_queue"]     else 0
            fps = step / max(time.time() - t0, 0.001)
            pct = 100 * step / total_steps
            print(f"  [{pct:5.1f}%] step={step:>5}  reward={r:+.2f}"
                  f"  queue={q:.2f}  ep={episodes}  fps={fps:.0f}")

        if step % save_ev == 0:
            save(os.path.join(config["paths"]["checkpoint_dir"], f"agent_step{step}.pt"))

    save(os.path.join(config["paths"]["checkpoint_dir"], "agent_final.pt"))
    np.save(os.path.join(config["paths"]["logs_dir"], "training_log.npy"), log)
    print("-" * 55)
    print(f"[Train] Finished. Episodes={episodes}  "
          f"Final mean reward={np.mean(log['episode_reward'] or [0]):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--steps",  type=int, default=500)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, total_steps=args.steps)
