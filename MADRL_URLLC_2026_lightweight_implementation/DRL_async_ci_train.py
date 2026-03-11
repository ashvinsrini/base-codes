
"""
Async 8-agent DRL pipeline converted from the notebook, with 10-run BLER CDF
confidence intervals.

Requirements already assumed available in your environment:
- env_orig.py containing class env
- LSTM.py containing class LSTMModel
- pretrained LSTM checkpoint (my path default: /home/sriniva3/Templates/LSTM_state_dict.pth, need to change this as per pathlocation)

This script does:
1) Loads the pretrained LSTM used for one-step SINR prediction.
2) Builds the wireless environment.
3) Trains the 8-agent async DRL system for NUM_RUNS independent runs.
4) Reproduces only the notebook's final BLER CDF plot, now with 95% CI.
5) Saves:
   - BLER CDF plot
   - raw BLER arrays per run
   - CI arrays needed to recreate the plot
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
from collections import deque
from pathlib import Path

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
from Utils.env_orig import env
from Utils.LSTM import LSTMModel  


# ============================================================
# User-configurable parameters
# ============================================================

OUTPUT_DIR = "async_8agent_bler_ci_outputs"
NUM_RUNS = 5
BASE_SEED = 42
FIX_ENVIRONMENT_ACROSS_RUNS = True

# LSTM checkpoint
LSTM_MODEL_DIR = Path("/home/sriniva3/Templates")
LSTM_MODEL_NAME = "LSTM_state_dict.pth"

# System dimensions
M, N, J = 8, 20, 3
Ts = 10000
p_o = 1e-5

# DRL training parameters (matching notebook)
T = 256
EPISODES = 20
BATCH_SIZE = 128
STATE_DIM = 2 * N * J
ACTION_DIM = N * J
MAX_ACTION = 1.0

# Plot / CI parameters
CI_XGRID_POINTS = 400
PLOT_FILENAME = "async_bler_cdf_ci.png"
RAW_BLER_FILENAME = "bler_runs_raw.npz"
CI_DATA_FILENAME = "bler_cdf_ci_data.npz"

# Devices
LSTM_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RL_DEVICE = torch.device("cpu")


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
# Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, next_state, action, reward):
        self.buffer.append((state, next_state, action, reward))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, next_state, action, reward = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(next_state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
        )

    def size(self):
        return len(self.buffer)


# ============================================================
# Helper functions from notebook
# ============================================================

def generate_matrix(N, J):
    matrix = np.zeros((N, J), dtype=int)

    for j in range(J):
        while True:
            row_index = np.random.randint(N)
            if np.sum(matrix[row_index]) == 0:
                matrix[row_index, j] = 1
                break

    for i in range(N):
        if np.sum(matrix[i]) == 0:
            if np.random.rand() > 0.5:
                col_index = np.random.randint(J)
                matrix[i, col_index] = 1

    return matrix


def return_bin_descrete_action(N, J, uhat):
    u = np.zeros((N, J))

    def select_unique_max_indices(matrix):
        selected_rows = set()
        result_indices = []

        for j in range(J):
            sorted_row_indices = np.argsort(matrix[:, j])[::-1]
            for row in sorted_row_indices:
                if row not in selected_rows:
                    selected_rows.add(row)
                    result_indices.append((row, j))
                    break
        return result_indices

    max_indices = select_unique_max_indices(uhat)

    allocated_resos = []
    for ind in max_indices:
        u[ind[0], ind[1]] = 1.0
        allocated_resos.extend([ind[0]])

    unallocated_resos = np.array([r for r in range(N) if r not in allocated_resos])
    v = np.concatenate((np.zeros((1, J)), np.eye(J)))

    for r in unallocated_resos:
        ind = np.argmin([np.linalg.norm(uhat[r, :] - v[n, :]) for n in range(v.shape[0])])
        u[r, :] = v[ind, :]

    return u


def return_cdf(a):
    sorted_a = np.sort(a)
    cdf = np.arange(1, len(sorted_a) + 1) / len(sorted_a)
    return sorted_a, cdf


# ============================================================
# Actor / Critic / DDPG
# ============================================================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x * self.max_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = F.relu(self.fc3(x))
        return q_value


class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(RL_DEVICE)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(RL_DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.5e-3)

        self.critic = Critic(state_dim, action_dim).to(RL_DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(RL_DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.5e-3)

        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(RL_DEVICE)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train(self, replay_buffer, t, batch_size=64):
        state, next_state, action, reward = replay_buffer.sample(batch_size)
        next_state = next_state.view(batch_size, -1).to(RL_DEVICE)
        state = state.view(batch_size, -1).to(RL_DEVICE)
        action = action.view(batch_size, -1).to(RL_DEVICE)
        reward = reward.to(RL_DEVICE)

        next_action = self.actor_target(next_state)
        target_q = self.critic_target(next_state, next_action)

        if t < T:
            not_done = 1.0
        else:
            not_done = 0.0

        target_q = reward + not_done * self.discount * target_q.detach()
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# ============================================================
# LSTM loading / prediction helper
# ============================================================

def load_pretrained_lstm():
    input_dim = N
    hidden_dim = 128
    output_dim = N
    num_layers = 1

    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(LSTM_DEVICE)
    lstm_ckpt_path = LSTM_MODEL_DIR / LSTM_MODEL_NAME

    if lstm_ckpt_path.exists():
        try:
            model.load_state_dict(torch.load(lstm_ckpt_path, map_location=LSTM_DEVICE))
            model.eval()
            print(f"Loaded pretrained LSTM model from: {lstm_ckpt_path.resolve()}")
        except Exception as e:
            print(f"Found '{lstm_ckpt_path}', but could not load it.")
            print(f"Error: {e}")
            print("Please run the LSTM_predict.py script first to save the model, and then run this pipeline again.")
            raise SystemExit(1)
    else:
        print(f"Pretrained LSTM model not found: {lstm_ckpt_path}")
        print("Please run the LSTM_predict.py script first to save the model, and then run this pipeline again.")
        raise SystemExit(1)

    return model


def get_lstm_pred_SINR(model, environ, alltime_PathGains, alltime_fast_fading_gains,
                       ts_start, ts_end, b=None, interfers_actions=None,
                       flag=None, b_actions=None):
    history = 10
    SINR_nxt_lstm = np.zeros((J, N), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for j in range(J):
            inp = []
            for ts in np.arange(ts_start, ts_end):
                SINR = environ.get_next_state(
                    alltime_PathGains,
                    alltime_fast_fading_gains,
                    ts=ts,
                    b=b,
                    interfers_actions=interfers_actions,
                    flag=flag,
                    b_actions=b_actions,
                )[0]
                inp.append(SINR[:, j])

            inp = np.array(inp, dtype=np.float32).reshape(1, history, N)
            pred = model(torch.tensor(inp, dtype=torch.float32, device=LSTM_DEVICE))
            SINR_nxt_lstm[j, :] = pred.cpu().numpy()[0]

    SINR_nxt_lstm = np.transpose(SINR_nxt_lstm)
    return SINR_nxt_lstm


# ============================================================
# Environment helpers
# ============================================================

def create_environment_and_channels(seed: int):
    set_all_seeds(seed)
    environ = env(Ts=Ts, N=N, M=M, J=J)
    alltime_fast_fading_gains, ff_gains = environ.fast_fading_channel_coefficients()
    TxRxds = environ.compute_TxRX()
    alltime_PathGains = environ.large_scale_fading_channel_coefficients(TxRxds)
    return environ, alltime_fast_fading_gains, ff_gains, TxRxds, alltime_PathGains


def build_interferers_actions_for_agent(agent_idx, joint_actions):
    return np.stack([joint_actions[k].T for k in range(M) if k != agent_idx], axis=0)


# ============================================================
# One full DRL run
# ============================================================

def run_one_training(realization, lstm_model, run_seed: int):
    environ, alltime_fast_fading_gains, _, _, alltime_PathGains = realization

    set_all_seeds(run_seed)

    agents = [DDPG(STATE_DIM, ACTION_DIM, MAX_ACTION) for _ in range(M)]
    replay_buffers = [ReplayBuffer() for _ in range(M)]

    epi_rewards = [[] for _ in range(M)]
    prev_actions = [generate_matrix(N, J) for _ in range(M)]
    ts_counter = 0
    time_slots = np.arange(0, Ts)

    for episode in tqdm(range(EPISODES), desc=f"Run seed {run_seed}", leave=False):
        states = []
        for k in range(M):
            interferers_prev = build_interferers_actions_for_agent(k, prev_actions)
            state_lstm_k = get_lstm_pred_SINR(
                model=lstm_model,
                environ=environ,
                alltime_PathGains=alltime_PathGains,
                alltime_fast_fading_gains=alltime_fast_fading_gains,
                ts_start=ts_counter,
                ts_end=ts_counter + 10,
                b=k,
                interfers_actions=interferers_prev,
                b_actions=prev_actions[k],
            )
            state_k = np.stack((state_lstm_k, prev_actions[k]), axis=0)
            states.append(state_k)

        episode_rewards = np.zeros(M, dtype=np.float32)

        for t in range(T):
            joint_actions = []
            for k in range(M):
                action_cont = agents[k].select_action(np.array(states[k]))
                action_disc = return_bin_descrete_action(N, J, np.reshape(action_cont, (N, J)))
                joint_actions.append(action_disc)

            next_states = []
            for k in range(M):
                interferers_k = build_interferers_actions_for_agent(k, joint_actions)

                next_state_lstm_k = get_lstm_pred_SINR(
                    model=lstm_model,
                    environ=environ,
                    alltime_PathGains=alltime_PathGains,
                    alltime_fast_fading_gains=alltime_fast_fading_gains,
                    ts_start=ts_counter + 1,
                    ts_end=ts_counter + 11,
                    b=k,
                    interfers_actions=interferers_k,
                    b_actions=joint_actions[k],
                )

                next_state_k = np.stack((next_state_lstm_k, joint_actions[k]), axis=0)

                _, _, reward_k = environ.compute_rewards(
                    alltime_PathGains,
                    alltime_fast_fading_gains,
                    ts=time_slots[ts_counter + 10],
                    b=k,
                    interfers_actions=interferers_k,
                    b_actions=joint_actions[k],
                )

                reward_k = np.sum(reward_k)
                episode_rewards[k] += reward_k

                replay_buffers[k].add(
                    states[k],
                    next_state_k,
                    joint_actions[k],
                    reward_k,
                )

                next_states.append(next_state_k)

            for k in range(M):
                if replay_buffers[k].size() > BATCH_SIZE:
                    agents[k].train(replay_buffers[k], t, BATCH_SIZE)

            states = next_states
            prev_actions = [a.copy() for a in joint_actions]

            ts_counter += 1
            if ts_counter >= Ts - 10:
                ts_counter = 0

        for k in range(M):
            epi_rewards[k].append(episode_rewards[k] / T)

    reward_df = pd.DataFrame({
        f"agent {k+1}": np.array(epi_rewards[k]) / (N * J)
        for k in range(M)
    })

    reward_smooth = reward_df.rolling(window=2).mean()
    err_prob_df = p_o * (10 ** (-reward_smooth))

    err_prob_matrix = err_prob_df.to_numpy()
    err_max = np.nanmax(err_prob_matrix, axis=1)
    err_max = np.asarray(err_max).reshape(-1)
    err_max = err_max[np.isfinite(err_max) & (err_max > 0)]

    cutoff = np.percentile(err_max, 10)
    err_max_main = err_max[err_max >= cutoff]
    return err_max_main.astype(np.float64)


# ============================================================
# CDF + CI processing
# ============================================================

def ecdf_on_grid(sorted_samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    return np.searchsorted(sorted_samples, x_grid, side="right") / len(sorted_samples)


def build_cdf_ci_from_runs(bler_runs, x_grid_points=400):
    mins = [np.min(a[a > 0]) for a in bler_runs if np.any(a > 0)]
    maxs = [np.max(a) for a in bler_runs]

    x_min = min(mins)
    x_max = max(maxs)
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

def save_bler_runs_npz(out_path: Path, bler_runs):
    save_dict = {f"run_{i+1:02d}": np.asarray(arr, dtype=np.float64) for i, arr in enumerate(bler_runs)}
    np.savez_compressed(out_path, **save_dict)


def plot_bler_cdf_ci(x_grid, cdf_mean, cdf_lo, cdf_hi, out_png: Path):
    plt.figure(figsize=[6, 4])
    plt.semilogx(x_grid, cdf_mean, linewidth=2.0, label="Async proposed approach BLER CDF")
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

    print(f"Using LSTM device: {LSTM_DEVICE}")
    print(f"Using RL device: {RL_DEVICE}")
    print(f"NUM_RUNS = {NUM_RUNS}")
    print(f"FIX_ENVIRONMENT_ACROSS_RUNS = {FIX_ENVIRONMENT_ACROSS_RUNS}")

    lstm_model = load_pretrained_lstm()

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

        bler_run = run_one_training(realization, lstm_model, run_seed)
        bler_runs.append(bler_run)
        print(f"Run {run_idx + 1}: collected {len(bler_run)} BLER points for final CDF")

    x_grid, cdf_runs, cdf_mean, cdf_lo, cdf_hi = build_cdf_ci_from_runs(
        bler_runs,
        x_grid_points=CI_XGRID_POINTS,
    )

    save_bler_runs_npz(out_dir / RAW_BLER_FILENAME, bler_runs)
    np.savez_compressed(
        out_dir / CI_DATA_FILENAME,
        x_grid=x_grid.astype(np.float64),
        cdf_runs=cdf_runs.astype(np.float64),
        cdf_mean=cdf_mean.astype(np.float64),
        cdf_lower=cdf_lo.astype(np.float64),
        cdf_upper=cdf_hi.astype(np.float64),
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
    print(" -", out_dir / RAW_BLER_FILENAME)
    print(" -", out_dir / CI_DATA_FILENAME)


if __name__ == "__main__":
    main()