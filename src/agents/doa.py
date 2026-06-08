"""
EdgeTransAI — Distributed Optimization Algorithm (DOA)
=======================================================
Implements the ADMM-adapted distributed coordination protocol that lets Edge
Coordination Agents (ECAs) synchronise their Traffic Signal Controller Agent
(TSCA) policies without a centralised parameter server.

Each ECA manages a region of M intersections.  Every T_sync milliseconds the
ECAs run one DOA iteration:
  1. Local gradient step on each TSCA's policy parameters
  2. Dual variable update (primal residual tracking)
  3. ECA aggregation (FedAvg-style global mean within region)
  4. Optional cross-region consensus (sparse graph message passing)

Convergence is monitored via the primal residual r^k = theta_i - theta_j.

Reference: Boyd et al. (2011), "Distributed Optimization via ADMM", FTML.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Lightweight snapshot of a single TSCA's relevant parameter block."""
    agent_id:  int
    theta:     np.ndarray        # flattened policy head parameters
    dual:      np.ndarray        # dual variable u_i
    region_id: int


@dataclass
class DOAMetrics:
    """Tracks convergence diagnostics across iterations."""
    primal_residuals: List[float] = field(default_factory=list)
    dual_residuals:   List[float] = field(default_factory=list)
    iterations:       int = 0
    converged:        bool = False
    wall_time_ms:     float = 0.0


# ---------------------------------------------------------------------------
# DOA coordinator
# ---------------------------------------------------------------------------

class DistributedOptimizationAlgorithm:
    """
    Coordinates policy synchronisation for one edge region.

    Parameters
    ----------
    region_id    : identifier for this ECA's region
    agent_ids    : list of TSCA IDs managed by this ECA
    param_dim    : dimensionality of the parameter vector to synchronise
                   (typically the final policy-head flattened weight)
    rho          : ADMM penalty coefficient
    eps_primal   : primal feasibility tolerance
    eps_dual     : dual convergence tolerance
    max_iters    : hard iteration cap
    """

    def __init__(
        self,
        region_id:   int,
        agent_ids:   List[int],
        param_dim:   int,
        rho:         float = 0.1,
        eps_primal:  float = 1e-3,
        eps_dual:    float = 1e-4,
        max_iters:   int   = 200,
    ):
        self.region_id  = region_id
        self.agent_ids  = agent_ids
        self.M          = len(agent_ids)
        self.param_dim  = param_dim
        self.rho        = rho
        self.eps_p      = eps_primal
        self.eps_d      = eps_dual
        self.max_iters  = max_iters

        # Local parameter copies and dual variables
        self.theta: Dict[int, np.ndarray] = {
            aid: np.zeros(param_dim) for aid in agent_ids
        }
        self.dual: Dict[int, np.ndarray] = {
            aid: np.zeros(param_dim) for aid in agent_ids
        }
        # Global consensus variable (ECA-level)
        self.theta_global = np.zeros(param_dim)
        self.metrics = DOAMetrics()

    # ------------------------------------------------------------------
    def receive_local_params(self, agent_id: int, theta: np.ndarray):
        """Called when a TSCA completes its local gradient step."""
        self.theta[agent_id] = theta.copy()

    # ------------------------------------------------------------------
    def step(self) -> Tuple[np.ndarray, DOAMetrics]:
        """
        Execute one DOA iteration.

        Returns
        -------
        theta_global : updated consensus parameter vector
        metrics      : convergence diagnostics
        """
        t0 = time.perf_counter()
        theta_prev_global = self.theta_global.copy()

        # Step 1: Update dual variables (primal residual)
        primal_res = []
        for aid in self.agent_ids:
            r = self.theta[aid] - self.theta_global
            self.dual[aid] = self.dual[aid] + r
            primal_res.append(np.linalg.norm(r))

        # Step 2: Global aggregation (FedAvg within region)
        stacked = np.stack([self.theta[aid] + self.dual[aid]
                            for aid in self.agent_ids], axis=0)
        self.theta_global = stacked.mean(axis=0)

        # Step 3: Dual residual (change in global consensus)
        dual_res = self.rho * np.linalg.norm(self.theta_global - theta_prev_global)

        primal_norm = float(np.max(primal_res))
        self.metrics.primal_residuals.append(primal_norm)
        self.metrics.dual_residuals.append(dual_res)
        self.metrics.iterations += 1
        self.metrics.wall_time_ms += (time.perf_counter() - t0) * 1000

        # Convergence check
        if primal_norm < self.eps_p and dual_res < self.eps_d:
            self.metrics.converged = True

        return self.theta_global.copy(), self.metrics

    # ------------------------------------------------------------------
    def broadcast_global(self) -> Dict[int, np.ndarray]:
        """
        Compute per-agent proximal update directions.
        Each TSCA receives: theta_global - dual_i (prox step signal).
        """
        return {
            aid: self.theta_global - self.dual[aid]
            for aid in self.agent_ids
        }

    # ------------------------------------------------------------------
    def reset_metrics(self):
        self.metrics = DOAMetrics()

    # ------------------------------------------------------------------
    @staticmethod
    def extract_param_vector(model: nn.Module, layer_name: str = "head") -> np.ndarray:
        """Extract a flattened parameter vector from a specific layer."""
        for name, param in model.named_parameters():
            if layer_name in name and "weight" in name:
                return param.detach().cpu().numpy().flatten()
        raise ValueError(f"Layer '{layer_name}' not found in model.")

    @staticmethod
    def inject_param_vector(model: nn.Module, vec: np.ndarray,
                             layer_name: str = "head"):
        """Inject a flat parameter vector back into a model layer."""
        for name, param in model.named_parameters():
            if layer_name in name and "weight" in name:
                target_shape = param.shape
                new_data = torch.tensor(
                    vec.reshape(target_shape), dtype=param.dtype
                )
                with torch.no_grad():
                    param.copy_(new_data)
                return
        raise ValueError(f"Layer '{layer_name}' not found in model.")


# ---------------------------------------------------------------------------
# City-level coordinator (multiple ECAs)
# ---------------------------------------------------------------------------

class CityCoordinator:
    """
    Manages multiple ECA regions and enables sparse cross-region consensus.
    In a production deployment this runs on each ECA; the inter-ECA
    communication is abstracted here for simulation purposes.
    """

    def __init__(self, config: dict):
        self.K = config["edge"]["num_nodes"]
        M = config["environment"]["num_intersections"]
        region_size = config["edge"]["region_size"]
        rho = config["doa"]["rho"]
        eps_p = config["doa"]["feasibility_tol"]
        eps_d = config["doa"]["convergence_tol"]
        max_it = config["doa"]["max_iterations"]

        # Estimate param_dim from config (action_dim * hidden[-1])
        param_dim = config["agent"]["action_dim"] * config["agent"]["hidden_dims"][-1]

        self.regions: List[DistributedOptimizationAlgorithm] = []
        for k in range(self.K):
            start = k * region_size
            end   = min(start + region_size, M)
            aids  = list(range(start, end))
            self.regions.append(
                DistributedOptimizationAlgorithm(k, aids, param_dim, rho, eps_p, eps_d, max_it)
            )

        # Adjacency for cross-region consensus (ring topology as default)
        self.neigh_map: Dict[int, List[int]] = {
            k: [(k - 1) % self.K, (k + 1) % self.K] for k in range(self.K)
        }

    # ------------------------------------------------------------------
    def synchronise_step(self, local_params: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        local_params : {agent_id: param_vector}
        Returns      : {agent_id: updated param_vector} after one DOA step.
        """
        # Feed local updates to each region
        for region in self.regions:
            for aid in region.agent_ids:
                if aid in local_params:
                    region.receive_local_params(aid, local_params[aid])

        # Run intra-region DOA step
        region_globals: Dict[int, np.ndarray] = {}
        for k, region in enumerate(self.regions):
            global_theta, _ = region.step()
            region_globals[k] = global_theta

        # Sparse cross-region consensus (one round)
        for k, region in enumerate(self.regions):
            neigh_thetas = [region_globals[j] for j in self.neigh_map[k]]
            if neigh_thetas:
                cross_mean = np.mean([region_globals[k]] + neigh_thetas, axis=0)
                region.theta_global = 0.9 * region.theta_global + 0.1 * cross_mean

        # Collect per-agent broadcast
        updates: Dict[int, np.ndarray] = {}
        for region in self.regions:
            updates.update(region.broadcast_global())
        return updates

    # ------------------------------------------------------------------
    def convergence_status(self) -> Dict[int, bool]:
        return {r.region_id: r.metrics.converged for r in self.regions}
