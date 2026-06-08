"""
EdgeTransAI — PPO Actor-Critic Agent
=====================================
Implements the Intent-Aware Multi-Agent Reinforcement Learning (IA-MARL)
controller using Proximal Policy Optimisation with Generalised Advantage
Estimation. Each traffic signal controller (TSCA) runs its own actor-critic
network; critics are augmented with neighbour observations (CTDE paradigm).

References
----------
Schulman et al. (2017) — Proximal Policy Optimization Algorithms
Lowe et al. (2017)     — MADDPG (centralised training, decentralised exec.)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Neural network blocks
# ---------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    """Lightweight attention over neighbour agent observations."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        query : (B, D)
        keys  : (B, N_neigh, D)
        """
        q = self.q(query).unsqueeze(1)                   # (B, 1, D)
        k = self.k(keys)                                  # (B, N, D)
        v = self.v(keys)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
        return torch.bmm(attn, v).squeeze(1)              # (B, D)


class ActorNetwork(nn.Module):
    """
    Decentralised actor — takes local observation and outputs a phase logit vector.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.head.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class CriticNetwork(nn.Module):
    """
    Centralised critic — receives local state + aggregated neighbour embedding.
    """
    def __init__(self, state_dim: int, hidden_dims: List[int], neigh_embed_dim: int = 64):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(state_dim, neigh_embed_dim), nn.ReLU())
        self.attn = SpatialAttention(neigh_embed_dim)
        in_dim = state_dim + neigh_embed_dim
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(in_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, state: torch.Tensor, neigh_states: torch.Tensor) -> torch.Tensor:
        """
        state        : (B, state_dim)
        neigh_states : (B, N_neigh, state_dim)
        """
        neigh_embed = self.embed(neigh_states)             # (B, N, D)
        query = self.embed(state)                          # (B, D)
        context = self.attn(query, neigh_embed)            # (B, D)
        combined = torch.cat([state, context], dim=-1)
        return self.value_head(self.trunk(combined))


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self, buffer_size: int, state_dim: int, n_agents: int, device: str):
        self.size = buffer_size
        self.device = device
        self.ptr = 0
        self.full = False

        self.states       = np.zeros((buffer_size, n_agents, state_dim), dtype=np.float32)
        self.actions      = np.zeros((buffer_size, n_agents), dtype=np.int64)
        self.log_probs    = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.rewards      = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.values       = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.dones        = np.zeros(buffer_size, dtype=np.float32)
        self.advantages   = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.returns      = np.zeros((buffer_size, n_agents), dtype=np.float32)

    def add(self, state, action, log_prob, reward, value, done):
        idx = self.ptr % self.size
        self.states[idx]    = state
        self.actions[idx]   = action
        self.log_probs[idx] = log_prob
        self.rewards[idx]   = reward
        self.values[idx]    = value
        self.dones[idx]     = float(done)
        self.ptr += 1
        if self.ptr >= self.size:
            self.full = True

    def compute_gae(self, last_values: np.ndarray, gamma: float, lam: float):
        """Compute Generalised Advantage Estimation in-place."""
        n = self.size if self.full else self.ptr
        gae = np.zeros_like(last_values)
        for t in reversed(range(n)):
            next_val = last_values if t == n - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, mini_batch_size: int):
        n = self.size if self.full else self.ptr
        indices = np.random.permutation(n)
        for start in range(0, n, mini_batch_size):
            idx = indices[start: start + mini_batch_size]
            yield (
                torch.tensor(self.states[idx], dtype=torch.float32),
                torch.tensor(self.actions[idx], dtype=torch.long),
                torch.tensor(self.log_probs[idx], dtype=torch.float32),
                torch.tensor(self.advantages[idx], dtype=torch.float32),
                torch.tensor(self.returns[idx], dtype=torch.float32),
            )

    def reset(self):
        self.ptr = 0
        self.full = False


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    Shared-weight PPO agent controlling all traffic signal agents.
    Each agent acts independently at inference (CTDE paradigm).

    Parameters
    ----------
    state_dim     : dimensionality of per-agent observation
    action_dim    : number of discrete signal phases
    config        : full config dict (agent, training sections)
    device        : 'cuda' | 'cpu'
    """

    def __init__(self, state_dim: int, action_dim: int, config: dict,
                 device: str = "cpu"):
        self.cfg = config["agent"]
        self.train_cfg = config["training"]
        self.device = torch.device(device)
        self.gamma = self.cfg["gamma"]
        self.lam = self.cfg["gae_lambda"]
        self.clip = self.cfg["clip_epsilon"]
        self.ent_c = self.cfg["entropy_coeff"]
        self.val_c = self.cfg["value_coeff"]
        self.max_grad = self.cfg["max_grad_norm"]

        hidden = self.cfg["hidden_dims"]
        self.actor  = ActorNetwork(state_dim, action_dim, hidden).to(self.device)
        self.critic = CriticNetwork(state_dim, hidden).to(self.device)

        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.cfg["learning_rate"]
        )
        self.buffer = RolloutBuffer(
            self.cfg["batch_size"], state_dim,
            config["environment"]["num_intersections"], device
        )

        # Logging
        self._loss_history: List[float] = []
        self._reward_history: List[float] = []
        self._step_count = 0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_actions(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        obs : (n_agents, state_dim)
        Returns actions, log_probs, values — all shape (n_agents,)
        """
        t_obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        logits = self.actor(t_obs)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        # Dummy neighbour states (zero) — replace with real neighbour obs in prod
        neigh = t_obs.unsqueeze(1).expand(-1, 1, -1)
        values = self.critic(t_obs, neigh).squeeze(-1)
        return (actions.cpu().numpy(),
                log_probs.cpu().numpy(),
                values.cpu().numpy())

    # ------------------------------------------------------------------
    def update(self, last_obs: np.ndarray) -> dict:
        """Run PPO update epochs. Returns dict of loss components."""
        t_last = torch.tensor(last_obs, dtype=torch.float32, device=self.device)
        neigh_last = t_last.unsqueeze(1).expand(-1, 1, -1)
        with torch.no_grad():
            last_vals = self.critic(t_last, neigh_last).squeeze(-1).cpu().numpy()
        self.buffer.compute_gae(last_vals, self.gamma, self.lam)

        total_loss_p, total_loss_v, total_ent = 0., 0., 0.
        num_updates = 0

        for _ in range(self.cfg["ppo_epochs"]):
            for (states, actions, old_lp, advantages, returns) in \
                    self.buffer.get_batches(self.cfg["mini_batch_size"]):
                states    = states.to(self.device)
                actions   = actions.to(self.device)
                old_lp    = old_lp.to(self.device)
                advantages = advantages.to(self.device)
                returns   = returns.to(self.device)

                # Flatten agent dimension for shared network
                B, N, D = states.shape
                s_flat = states.view(B * N, D)
                a_flat = actions.view(B * N)
                lp_old_flat = old_lp.view(B * N)
                adv_flat = advantages.view(B * N)
                ret_flat = returns.view(B * N)

                # Normalise advantages
                adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

                # Policy loss
                logits = self.actor(s_flat)
                dist = Categorical(logits=logits)
                lp_new = dist.log_prob(a_flat)
                ratio = torch.exp(lp_new - lp_old_flat)
                loss_p = -torch.min(
                    ratio * adv_flat,
                    torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_flat
                ).mean()

                # Value loss
                neigh_flat = s_flat.unsqueeze(1)
                values_pred = self.critic(s_flat, neigh_flat).squeeze(-1)
                loss_v = F.mse_loss(values_pred, ret_flat)

                # Entropy bonus
                entropy = dist.entropy().mean()

                loss = loss_p + self.val_c * loss_v - self.ent_c * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad
                )
                self.opt.step()

                total_loss_p += loss_p.item()
                total_loss_v += loss_v.item()
                total_ent    += entropy.item()
                num_updates  += 1

        self.buffer.reset()
        return {
            "loss_policy": total_loss_p / max(num_updates, 1),
            "loss_value":  total_loss_v / max(num_updates, 1),
            "entropy":     total_ent    / max(num_updates, 1),
        }

    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "opt":    self.opt.state_dict(),
            "step":   self._step_count,
        }, path)
        print(f"[PPOAgent] Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.opt.load_state_dict(ckpt["opt"])
        self._step_count = ckpt.get("step", 0)
        print(f"[PPOAgent] Checkpoint loaded ← {path}")
