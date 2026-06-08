"""
EdgeTransAI — Traffic Environment
==================================
Wraps a SUMO simulation (or a lightweight numpy-based stub when SUMO is not
installed) to expose a standard Gymnasium-like multi-agent interface.

Usage
-----
    from src.simulation.traffic_env import TrafficEnv
    env = TrafficEnv(config)
    obs = env.reset()
    obs, rewards, dones, info = env.step(actions)
"""

import os
import sys
import time
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# SUMO detection — fall back to numpy stub if SUMO is not installed
# ---------------------------------------------------------------------------
SUMO_AVAILABLE = False
try:
    if "SUMO_HOME" in os.environ:
        sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
    import traci  # type: ignore
    SUMO_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Lightweight stub used when SUMO is absent (useful for unit-testing models)
# ---------------------------------------------------------------------------
class _SUMOStub:
    """Synthetic traffic simulator — reproduces realistic queue / speed dynamics
    using a stochastic CTMC-inspired model without requiring a SUMO install."""

    def __init__(self, num_intersections: int, seed: int = 42):
        self.n = num_intersections
        self.rng = np.random.default_rng(seed)
        self._step = 0
        self._queues = np.zeros((self.n, 4))      # lanes per intersection
        self._speeds = np.ones((self.n, 4)) * 50.0
        self._phases = np.zeros(self.n, dtype=int)
        self._phase_elapsed = np.zeros(self.n)
        self._waiting = np.zeros(self.n)

    def reset(self):
        self._step = 0
        self._queues = self.rng.uniform(0, 5, (self.n, 4))
        self._speeds = self.rng.uniform(30, 60, (self.n, 4))
        self._phases[:] = 0
        self._phase_elapsed[:] = 0
        self._waiting[:] = 0

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Advance one simulation step.  Returns updated queues, speeds, total_delay."""
        self._step += 1
        hour_fraction = (self._step * 5) / 3600.0  # 5-second steps → hour
        # Bimodal demand curve (morning / evening peaks)
        demand = (0.3
                  + 0.5 * np.exp(-((hour_fraction - 8) ** 2) / 4)
                  + 0.4 * np.exp(-((hour_fraction - 17.5) ** 2) / 3))
        demand = float(np.clip(demand, 0.1, 1.0))

        for i in range(self.n):
            # Arrival
            arrivals = self.rng.poisson(demand * 3, 4).astype(float)
            # Service: depends on green phase (simplified — 2 phases served simultaneously)
            phase = int(actions[i]) % 8
            served_lanes = {
                0: [0, 2], 1: [0, 2], 2: [1, 3], 3: [1, 3],
                4: [0],    5: [2],    6: [1],    7: [3],
            }[phase]
            service = np.zeros(4)
            for l in served_lanes:
                service[l] = self.rng.uniform(2, 4)

            phase_changed = (phase != self._phases[i])
            self._queues[i] = np.clip(self._queues[i] + arrivals - service, 0, 50)
            self._speeds[i] = np.clip(
                60 - self._queues[i] * 1.2 + self.rng.normal(0, 2, 4), 5, 80
            )
            self._phases[i] = phase
            self._phase_elapsed[i] = 0 if phase_changed else self._phase_elapsed[i] + 1
            self._waiting[i] = float(np.max(self._queues[i]))

        total_delay = float(np.sum(self._queues))
        return self._queues.copy(), self._speeds.copy(), total_delay


# ---------------------------------------------------------------------------
# Main multi-agent environment
# ---------------------------------------------------------------------------
class TrafficEnv:
    """
    Multi-agent traffic signal control environment.

    Observation per agent (TSCA):
      - Queue lengths per lane  : shape (4, 8)  — 8 discretised cells
      - Average approach speed  : shape (4,)
      - Current signal phase    : shape (8,)    — one-hot
      - Phase elapsed time      : scalar
      - Intent distribution     : shape (7,)    — upstream N(i) avg

    Action space (per agent):
      - Integer in {0, …, 7}   — selected next signal phase
    """

    def __init__(self, config: dict, intent_model=None):
        self.cfg = config
        self.n_agents = config["environment"]["num_intersections"]
        self.num_phases = config["environment"]["num_phases"]
        self.min_green = config["environment"]["min_green"]
        self.max_green = config["environment"]["max_green"]
        self.time_step = config["environment"]["time_step"]
        self.max_steps = config["environment"]["simulation_steps"] // self.time_step

        self.state_dim = config["agent"]["state_dim"]
        self.action_dim = config["agent"]["action_dim"]

        self._sim = _SUMOStub(self.n_agents, seed=config["training"]["seed"])
        self._step_count = 0
        self._phase_elapsed = np.zeros(self.n_agents)
        self._current_phases = np.zeros(self.n_agents, dtype=int)
        self._queues = np.zeros((self.n_agents, 4))
        self._speeds = np.ones((self.n_agents, 4)) * 50.0

        # --- BiLSTM Intent Model ---
        self.intent_dim = config["intent_model"]["num_classes"]   # 4
        self._intents = np.ones((self.n_agents, self.intent_dim)) / self.intent_dim

        self._intent_model = None
        if intent_model is False:
            # Explicitly disabled — no model, no disk load
            pass
        elif intent_model is not None:
            # Use injected model (avoids repeated disk I/O in multi-env runs)
            self._intent_model = intent_model
        else:
            intent_ckpt = os.path.join(
                config.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints"),
                "intent_best.pt"
            )
            if os.path.exists(intent_ckpt):
                from src.models.intent_model import IntentModel
                self._intent_model = IntentModel(config)
                self._intent_model.load(intent_ckpt)
                self._intent_model.eval()
                print(f"[TrafficEnv] BiLSTM intent model loaded from {intent_ckpt}")
            else:
                print(f"[TrafficEnv] WARNING: No intent checkpoint at {intent_ckpt}. "
                      f"Using uniform intent. Run: python -m src.models.intent_model --train")

        # Trajectory buffer: last seq_len position-velocity readings per agent
        self._seq_len = config["intent_model"]["sequence_length"]   # 10
        # Shape: (n_agents, seq_len, 4) — [x, y, vx, vy]
        self._traj_buf = np.zeros((self.n_agents, self._seq_len, 4), dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self._sim.reset()
        self._step_count = 0
        self._phase_elapsed[:] = 0
        self._current_phases[:] = 0
        self._queues, self._speeds, _ = self._sim.step(self._current_phases)
        self._intents = np.ones((self.n_agents, self.intent_dim)) / self.intent_dim
        self._traj_buf = np.zeros((self.n_agents, self._seq_len, 4), dtype=np.float32)
        return self._build_obs()

    # ------------------------------------------------------------------
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
        """
        actions : np.ndarray of shape (n_agents,), dtype int

        Returns
        -------
        obs     : np.ndarray (n_agents, state_dim)
        rewards : np.ndarray (n_agents,)
        done    : bool
        info    : dict
        """
        # Enforce min-green constraint
        safe_actions = actions.copy()
        for i in range(self.n_agents):
            if self._phase_elapsed[i] < self.min_green:
                safe_actions[i] = self._current_phases[i]

        prev_queues = self._queues.copy()
        self._queues, self._speeds, total_delay = self._sim.step(safe_actions)

        # Update elapsed counters
        for i in range(self.n_agents):
            if safe_actions[i] != self._current_phases[i]:
                self._phase_elapsed[i] = 0
            else:
                self._phase_elapsed[i] += 1
        self._current_phases = safe_actions.copy()

        # Simulate intent update (in production this comes from BiLSTM)
        self._intents = self._update_intents()

        rewards = self._compute_rewards(prev_queues, actions, safe_actions)
        self._step_count += 1
        done = self._step_count >= self.max_steps

        info = {
            "total_delay": total_delay,
            "mean_queue": float(np.mean(self._queues)),
            "step": self._step_count,
        }
        return self._build_obs(), rewards, done, info

    # ------------------------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        obs = []
        for i in range(self.n_agents):
            # Discretise queue into 8 cells per lane (occupied=1, empty=0)
            queue_enc = np.zeros(32)
            for l in range(4):
                cells_filled = min(int(self._queues[i, l]), 8)
                queue_enc[l * 8: l * 8 + cells_filled] = 1.0
            speed_norm = self._speeds[i] / 80.0                    # (4,)
            phase_onehot = np.eye(8)[self._current_phases[i]]      # (8,)
            elapsed_norm = np.array([min(self._phase_elapsed[i] / 90.0, 1.0)])  # (1,)
            intent = self._intents[i]                               # (4,)
            # Total: 32 + 4 + 8 + 1 + 4 = 49
            obs.append(np.concatenate([queue_enc, speed_norm, phase_onehot,
                                        elapsed_norm, intent]))
        return np.array(obs, dtype=np.float32)  # (n_agents, 49)

    # ------------------------------------------------------------------
    def _compute_rewards(self, prev_q, actions, safe_actions):
        alpha = self.cfg["reward"]["alpha"]
        beta = self.cfg["reward"]["beta"]
        gamma_r = self.cfg["reward"]["gamma"]
        rewards = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            queue_pen = -alpha * float(np.sum(self._queues[i]))
            wait_pen = -beta * float(np.max(self._queues[i]))
            change_pen = -gamma_r * float(actions[i] != self._current_phases[i])
            rewards[i] = queue_pen + wait_pen + change_pen
        return rewards

    # ------------------------------------------------------------------
    def _update_intents(self) -> np.ndarray:
        """
        Run BiLSTM inference on each agent's trajectory buffer.
        Falls back to uniform distribution if model is not loaded.
        """
        if self._intent_model is None:
            # Fallback: uniform distribution
            return np.ones((self.n_agents, self.intent_dim)) / self.intent_dim

        # Build synthetic trajectory from queue/speed as [x, y, vx, vy] proxy.
        # In a real deployment this comes from GPS probe vehicles.
        # Here we derive a plausible approach trajectory from simulation state.
        for i in range(self.n_agents):
            mean_speed = float(np.mean(self._speeds[i]))   # km/h
            mean_queue = float(np.mean(self._queues[i]))

            # Simulate an approaching vehicle's last seq_len positions
            # Deceleration signature encodes turning intent
            speed_ms = mean_speed / 3.6
            # Vehicles with high queues on side lanes → more likely turning
            left_queue  = self._queues[i, 1]
            right_queue = self._queues[i, 3]
            # Shift trajectory buffer (slide window forward)
            self._traj_buf[i, :-1] = self._traj_buf[i, 1:]
            x_prev = self._traj_buf[i, -2, 0] if self._step_count > 1 else 0.0
            y_prev = self._traj_buf[i, -2, 1] if self._step_count > 1 else 0.0
            # Direction hint from queue imbalance
            heading_bias = (right_queue - left_queue) * 0.05
            vx = speed_ms * np.cos(heading_bias) + np.random.normal(0, 0.1)
            vy = speed_ms * np.sin(heading_bias) + np.random.normal(0, 0.1)
            self._traj_buf[i, -1] = [x_prev + vx * 0.1, y_prev + vy * 0.1, vx, vy]

        # Run BiLSTM on all agents in one batched forward pass
        with torch.no_grad():
            traj_tensor = torch.tensor(self._traj_buf, dtype=torch.float32)
            import torch.nn.functional as F
            logits = self._intent_model(traj_tensor)          # (n_agents, 4)
            probs  = F.softmax(logits, dim=-1).numpy()        # (n_agents, 4)

        return probs

    # ------------------------------------------------------------------
    @property
    def num_agents(self):
        return self.n_agents
