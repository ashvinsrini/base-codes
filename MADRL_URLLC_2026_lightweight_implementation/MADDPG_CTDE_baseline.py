"""
MADDPG CTDE baseline converted to a standalone Python script.

This script:
1) Loads the MADDPG CTDE baseline setup.
2) Runs NUM_RUNS independent training runs.
3) Uses 20 episodes per run.
4) Reproduces only the final BLER CDF with 95% CI across runs.
5) Saves only:
   - MADDPG BLER CDF plot (.png)
   - CDF/CI data used to plot that figure (.npz)



Required file in your working tree:
- ./Utils/env_orig_maddpg.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import random
from collections import deque
from pathlib import Path
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


# ============================================================
# User-configurable parameters
# ============================================================

OUTPUT_DIR = "maddpg_ctde_bler_ci_outputs"

NUM_RUNS = 5
BASE_SEED = 42
FIX_ENVIRONMENT_ACROSS_RUNS = True

# System dimensions
M = 8
N = 20
J = 3
P_O = 1e-5

TS = 10_000
EPISODES = 20
HORIZON_T = 256
BATCH_SIZE = 128
REPLAY_SIZE = int(1e5)

ACTOR_LR = 5e-4
CRITIC_LR = 5e-4
GAMMA = 0.99
TAU = 0.005

ACTOR_HIDDEN = [256, 128, 64]
CRITIC_HIDDEN = [512, 256, 128]

CI_XGRID_POINTS = 400
PLOT_FILENAME = "maddpg_ctde_bler_cdf_ci.png"
CI_DATA_FILENAME = "maddpg_ctde_bler_cdf_ci_data.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Utilities
# ============================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def t_critical_95(n_runs: int) -> float:
    table = {
        2: 12.706204736432095,
        3: 4.302652729696142,
        4: 3.182446305284263,
        5: 2.7764451051977987,
        6: 2.5705818366147395,
        7: 2.4469118511449692,
        8: 2.3646242515927844,
        9: 2.306004135204166,
        10: 2.2621571628540993,
    }
    return table.get(n_runs, 1.96)


# ============================================================
# Load environment module from notebook path
# ============================================================

ENV_PATH = Path("./Utils/env_orig_maddpg.py")
if not ENV_PATH.exists():
    raise FileNotFoundError(
        "Could not find ./Utils/env_orig_maddpg.py. "
        "Place env_orig_maddpg.py inside the Utils folder."
    )

spec = importlib.util.spec_from_file_location("env_orig_maddpg", ENV_PATH)
env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env_module)
env = env_module.env


# ============================================================
# Notebook helper functions
# ============================================================

def generate_matrix(num_subchannels: int, num_devices: int) -> np.ndarray:
    matrix = np.zeros((num_subchannels, num_devices), dtype=np.float32)

    for j in range(num_devices):
        while True:
            row_index = np.random.randint(num_subchannels)
            if np.sum(matrix[row_index]) == 0:
                matrix[row_index, j] = 1.0
                break

    for i in range(num_subchannels):
        if np.sum(matrix[i]) == 0 and np.random.rand() > 0.5:
            col_index = np.random.randint(num_devices)
            matrix[i, col_index] = 1.0

    return matrix


def build_interferers_actions_for_agent(agent_idx: int, joint_actions):
    return np.stack(
        [joint_actions[k].T for k in range(len(joint_actions)) if k != agent_idx],
        axis=0
    )


def get_local_state(environ,
                    alltime_PathGains,
                    alltime_fast_fading_gains,
                    ts: int,
                    agent_idx: int,
                    prev_actions):
    interferers = build_interferers_actions_for_agent(agent_idx, prev_actions)
    state = environ.get_next_state(
        alltime_PathGains,
        alltime_fast_fading_gains,
        ts=ts,
        b=agent_idx,
        interfers_actions=interferers,
        b_actions=prev_actions[agent_idx]
    )
    return state.astype(np.float32)


def select_unique_max_indices(matrix: np.ndarray):
    selected_rows = set()
    result_indices = []

    for j in range(matrix.shape[1]):
        sorted_row_indices = np.argsort(matrix[:, j])[::-1]
        for row in sorted_row_indices:
            if row not in selected_rows:
                selected_rows.add(int(row))
                result_indices.append((int(row), j))
                break

    return result_indices


def quantize_proto_action_urlcc(proto_action: np.ndarray,
                                num_subchannels: int,
                                num_devices: int) -> np.ndarray:
    uhat = np.asarray(proto_action).reshape(num_subchannels, num_devices)
    u = np.zeros((num_subchannels, num_devices), dtype=np.float32)

    max_indices = select_unique_max_indices(uhat)
    allocated_resources = []

    for row_idx, col_idx in max_indices:
        u[row_idx, col_idx] = 1.0
        allocated_resources.append(row_idx)

    unallocated_resources = [r for r in range(num_subchannels) if r not in allocated_resources]
    v = np.concatenate((np.zeros((1, num_devices)), np.eye(num_devices)), axis=0)

    for r in unallocated_resources:
        ind = np.argmin([np.linalg.norm(uhat[r, :] - v[n, :]) for n in range(v.shape[0])])
        u[r, :] = v[ind, :]

    return u


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def add(self, local_states, joint_state, next_local_states, next_joint_state, joint_action, rewards):
        self.buffer.append(
            (local_states, joint_state, next_local_states, next_joint_state, joint_action, rewards)
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        local_states, joint_states, next_local_states, next_joint_states, joint_actions, rewards = zip(*batch)

        return (
            torch.tensor(np.stack(local_states), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.stack(joint_states), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.stack(next_local_states), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.stack(next_joint_states), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.stack(joint_actions), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.stack(rewards), dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Networks
# ============================================================

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[2], action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))


class CentralizedCritic(nn.Module):
    def __init__(self, joint_state_dim: int, joint_action_dim: int, hidden_dims):
        super().__init__()
        self.fc1 = nn.Linear(joint_state_dim + joint_action_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[2], 1)

    def forward(self, joint_state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([joint_state, joint_action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class MADDPGAgent:
    def __init__(self, state_dim: int, action_dim: int, joint_state_dim: int, joint_action_dim: int):
        self.actor = Actor(state_dim, action_dim, ACTOR_HIDDEN).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim, ACTOR_HIDDEN).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CentralizedCritic(joint_state_dim, joint_action_dim, CRITIC_HIDDEN).to(DEVICE)
        self.critic_target = CentralizedCritic(joint_state_dim, joint_action_dim, CRITIC_HIDDEN).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

    def select_action(self, state: np.ndarray, exploration_std: float) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32, device=DEVICE)
            proto_action = self.actor(state_tensor).cpu().numpy().flatten()

        noise = np.random.normal(0.0, exploration_std, size=proto_action.shape)
        proto_action = np.clip(proto_action + noise, 0.0, 1.0)
        return proto_action


def soft_update(source: nn.Module, target: nn.Module, tau: float):
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.copy_(tau * src_param.data + (1.0 - tau) * tgt_param.data)


def train_one_step(agents, replay_buffer, state_shape, action_shape):
    if len(replay_buffer) < BATCH_SIZE:
        return

    local_states, joint_states, next_local_states, next_joint_states, joint_actions, rewards = replay_buffer.sample(BATCH_SIZE)

    batch_size = local_states.shape[0]
    per_agent_state_dim = int(np.prod(state_shape))
    per_agent_action_dim = int(np.prod(action_shape))

    joint_states_flat = joint_states.view(batch_size, -1)
    next_joint_states_flat = next_joint_states.view(batch_size, -1)
    joint_actions_flat = joint_actions.view(batch_size, -1)

    for agent_idx, agent in enumerate(agents):
        with torch.no_grad():
            next_actions_all = []
            for other_idx, other_agent in enumerate(agents):
                next_local_state = next_local_states[:, other_idx].contiguous().view(batch_size, per_agent_state_dim)
                next_action = other_agent.actor_target(next_local_state)
                next_actions_all.append(next_action.view(batch_size, per_agent_action_dim))
            next_joint_actions_flat = torch.cat(next_actions_all, dim=-1)

            target_q = agent.critic_target(next_joint_states_flat, next_joint_actions_flat)
            y = rewards[:, agent_idx:agent_idx + 1] + GAMMA * target_q

        current_q = agent.critic(joint_states_flat, joint_actions_flat)
        critic_loss = F.mse_loss(current_q, y)

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        pred_actions_all = []
        for other_idx, other_agent in enumerate(agents):
            local_state = local_states[:, other_idx].contiguous().view(batch_size, per_agent_state_dim)
            if other_idx == agent_idx:
                pred_action = other_agent.actor(local_state)
            else:
                pred_action = other_agent.actor(local_state).detach()
            pred_actions_all.append(pred_action.view(batch_size, per_agent_action_dim))

        pred_joint_actions_flat = torch.cat(pred_actions_all, dim=-1)
        actor_loss = -agent.critic(joint_states_flat, pred_joint_actions_flat).mean()

        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        soft_update(agent.actor, agent.actor_target, TAU)
        soft_update(agent.critic, agent.critic_target, TAU)


# ============================================================
# Environment helpers
# ============================================================

def create_environment_and_channels(seed: int):
    set_all_seeds(seed)
    environ = env(Ts=TS, N=N, M=M, J=J)
    alltime_fast_fading_gains, ff_gains = environ.fast_fading_channel_coefficients()
    TxRxds = environ.compute_TxRX()
    alltime_PathGains = environ.large_scale_fading_channel_coefficients(TxRxds)
    return environ, alltime_fast_fading_gains, ff_gains, TxRxds, alltime_PathGains


# ============================================================
# One training run
# ============================================================

def run_one_training(realization, run_seed: int) -> np.ndarray:
    environ, alltime_fast_fading_gains, _, _, alltime_PathGains = realization

    set_all_seeds(run_seed)

    state_shape = (2, N, J)
    action_shape = (N, J)

    state_dim = int(np.prod(state_shape))
    action_dim = int(np.prod(action_shape))
    joint_state_dim = M * state_dim
    joint_action_dim = M * action_dim

    agents = [MADDPGAgent(state_dim, action_dim, joint_state_dim, joint_action_dim) for _ in range(M)]
    replay_buffer = ReplayBuffer(REPLAY_SIZE)

    prev_actions = [generate_matrix(N, J) for _ in range(M)]
    ts_counter = 1

    episode_rewards = [[] for _ in range(M)]

    for episode in tqdm(range(EPISODES), desc=f"Run seed {run_seed}", leave=False):
        local_states = [
            get_local_state(
                environ,
                alltime_PathGains,
                alltime_fast_fading_gains,
                ts_counter,
                agent_idx,
                prev_actions
            )
            for agent_idx in range(M)
        ]

        rewards_this_episode = np.zeros(M, dtype=np.float32)
        exploration_std = max(1.0 / np.sqrt(episode + 1), 1e-2)

        for _ in range(HORIZON_T):
            joint_state = np.stack(local_states, axis=0)

            joint_actions = []
            for agent_idx in range(M):
                proto_action = agents[agent_idx].select_action(local_states[agent_idx], exploration_std)
                discrete_action = quantize_proto_action_urlcc(proto_action, N, J)
                joint_actions.append(discrete_action)

            next_local_states = []
            step_rewards = np.zeros(M, dtype=np.float32)

            for agent_idx in range(M):
                interferers_actions = build_interferers_actions_for_agent(agent_idx, joint_actions)

                _, _, reward_agent = environ.compute_rewards(
                    alltime_PathGains,
                    alltime_fast_fading_gains,
                    ts=min(ts_counter + 1, TS - 1),
                    b=agent_idx,
                    interfers_actions=interferers_actions,
                    b_actions=joint_actions[agent_idx]
                )

                next_state_agent = environ.get_next_state(
                    alltime_PathGains,
                    alltime_fast_fading_gains,
                    ts=min(ts_counter + 1, TS - 1),
                    b=agent_idx,
                    interfers_actions=interferers_actions,
                    b_actions=joint_actions[agent_idx]
                ).astype(np.float32)

                next_local_states.append(next_state_agent)
                step_rewards[agent_idx] = float(np.sum(reward_agent))
                rewards_this_episode[agent_idx] += step_rewards[agent_idx]

            replay_buffer.add(
                np.stack(local_states, axis=0),
                joint_state,
                np.stack(next_local_states, axis=0),
                np.stack(next_local_states, axis=0),
                np.stack(joint_actions, axis=0),
                step_rewards.astype(np.float32),
            )

            train_one_step(agents, replay_buffer, state_shape, action_shape)

            local_states = next_local_states
            prev_actions = [a.copy() for a in joint_actions]

            ts_counter += 1
            if ts_counter >= TS - 2:
                ts_counter = 1

        for agent_idx in range(M):
            episode_rewards[agent_idx].append(rewards_this_episode[agent_idx] / HORIZON_T)

    reward_df = pd.DataFrame({
        f"agent_{k+1}": np.array(episode_rewards[k]) / (N * J)
        for k in range(M)
    })

    reward_smooth = reward_df.rolling(window=1).mean()

    err_prob_df = P_O * (10 ** (-reward_smooth))
    err_prob_matrix = err_prob_df.to_numpy()

    err_max = np.nanmax(err_prob_matrix, axis=1)
    err_max = np.asarray(err_max).reshape(-1)
    err_max = err_max[np.isfinite(err_max) & (err_max > 0)]

    if err_max.size == 0:
        raise RuntimeError("No valid BLER samples were produced in this run.")

    cutoff = np.percentile(err_max, 10)
    err_max_main = err_max[err_max >= cutoff]

    if err_max_main.size == 0:
        err_max_main = err_max.copy()

    return err_max_main.astype(np.float64)


# ============================================================
# CDF + CI processing
# ============================================================

def ecdf_on_grid(sorted_samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    return np.searchsorted(sorted_samples, x_grid, side="right") / len(sorted_samples)


def build_cdf_ci_from_runs(bler_runs, x_grid_points=400):
    mins = [np.min(a[a > 0]) for a in bler_runs if np.any(a > 0)]
    maxs = [np.max(a) for a in bler_runs]

    if len(mins) == 0 or len(maxs) == 0:
        raise RuntimeError("Could not build CDF because no positive BLER samples were found.")

    x_min = min(mins)
    x_max = max(maxs)

    if np.isclose(x_min, x_max):
        x_min *= 0.95
        x_max *= 1.05

    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), x_grid_points)

    cdf_runs = []
    for arr in bler_runs:
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        arr = np.sort(arr)
        cdf_runs.append(ecdf_on_grid(arr, x_grid))

    cdf_runs = np.stack(cdf_runs, axis=0)
    cdf_mean = np.mean(cdf_runs, axis=0)

    if cdf_runs.shape[0] == 1:
        cdf_lo = cdf_mean.copy()
        cdf_hi = cdf_mean.copy()
    else:
        std = np.std(cdf_runs, axis=0, ddof=1)
        se = std / np.sqrt(cdf_runs.shape[0])
        crit = t_critical_95(cdf_runs.shape[0])
        half = crit * se
        cdf_lo = np.clip(cdf_mean - half, 0.0, 1.0)
        cdf_hi = np.clip(cdf_mean + half, 0.0, 1.0)

    return x_grid, cdf_runs, cdf_mean, cdf_lo, cdf_hi


# ============================================================
# Plotting / saving
# ============================================================

def save_cdf_ci_npz(out_path: Path, x_grid, cdf_runs, cdf_mean, cdf_lo, cdf_hi):
    np.savez_compressed(
        out_path,
        x_grid=np.asarray(x_grid, dtype=np.float64),
        cdf_runs=np.asarray(cdf_runs, dtype=np.float64),
        cdf_mean=np.asarray(cdf_mean, dtype=np.float64),
        cdf_lower=np.asarray(cdf_lo, dtype=np.float64),
        cdf_upper=np.asarray(cdf_hi, dtype=np.float64),
    )


def plot_bler_cdf_ci(x_grid, cdf_mean, cdf_lo, cdf_hi, out_png: Path):
    plt.figure(figsize=(6, 4))
    plt.semilogx(x_grid, cdf_mean, linewidth=2.0, label="MADDPG CTDE baseline BLER CDF")
    plt.fill_between(x_grid, cdf_lo, cdf_hi, alpha=0.20, label="95% CI")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.xlabel("Err_prob")
    plt.ylabel("CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"NUM_RUNS = {NUM_RUNS}")
    print(f"EPISODES = {EPISODES}")
    print(f"HORIZON_T = {HORIZON_T}")
    print(f"FIX_ENVIRONMENT_ACROSS_RUNS = {FIX_ENVIRONMENT_ACROSS_RUNS}")

    if FIX_ENVIRONMENT_ACROSS_RUNS:
        shared_realization = create_environment_and_channels(BASE_SEED)
    else:
        shared_realization = None

    bler_runs = []

    for run_idx in range(NUM_RUNS):
        run_seed = BASE_SEED + 1000 + run_idx
        print(f"\nStarting run {run_idx + 1}/{NUM_RUNS} with seed {run_seed}")

        if FIX_ENVIRONMENT_ACROSS_RUNS:
            realization = shared_realization
        else:
            realization = create_environment_and_channels(run_seed)

        bler_run = run_one_training(realization, run_seed)
        bler_runs.append(bler_run)

        print(f"Run {run_idx + 1}: collected {len(bler_run)} BLER points for final CDF")

    x_grid, cdf_runs, cdf_mean, cdf_lo, cdf_hi = build_cdf_ci_from_runs(
        bler_runs=bler_runs,
        x_grid_points=CI_XGRID_POINTS,
    )

    save_cdf_ci_npz(
        out_dir / CI_DATA_FILENAME,
        x_grid=x_grid,
        cdf_runs=cdf_runs,
        cdf_mean=cdf_mean,
        cdf_lo=cdf_lo,
        cdf_hi=cdf_hi,
    )

    plot_bler_cdf_ci(
        x_grid=x_grid,
        cdf_mean=cdf_mean,
        cdf_lo=cdf_lo,
        cdf_hi=cdf_hi,
        out_png=out_dir / PLOT_FILENAME,
    )

    print("\nSaved outputs in:", out_dir.resolve())
    print(" -", out_dir / PLOT_FILENAME)
    print(" -", out_dir / CI_DATA_FILENAME)


if __name__ == "__main__":
    main()