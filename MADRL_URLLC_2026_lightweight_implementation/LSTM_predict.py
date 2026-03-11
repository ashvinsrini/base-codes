#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Notebook-matching LSTM interference script with 95% confidence intervals.

What this fixes compared to the previous 5-run script:
1) Uses NUM_EPOCHS = 50 to match the notebook.
2) Records training loss per mini-batch update (not per epoch average),
   so the loss curve matches the notebook-style long semilogy plot.
3) Uses the full test trace and x-axis scaling xticks/10, matching the notebook.
4) Keeps prediction strictly one-step-ahead (no recursive rollout).
5) Uses the same fixed simulation across runs so the target trace stays identical
   and the CI only reflects training randomness.

Outputs saved:
- training_mse_loss_ci.png / .pdf
- interference_trace_ci.png / .pdf
- raw numpy arrays and metadata
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm


# ============================================================
# User-configurable parameters
# ============================================================

OUTPUT_DIR = "lstm_interfere_ci_fixed_outputs"
MODEL_SAVE_NAME = "LSTM_state_dict.pth"
SAVE_BEST_MODEL = True
NUM_RUNS = 5
NUM_EPOCHS = 10
BASE_SEED = 12345

# Keep the simulation fixed across runs so that:
# - the target trace is the same
# - CI reflects only training randomness
FIX_SIMULATION_ACROSS_RUNS = True

# Simulation parameters matching the notebook
Pt_dBm = 20
k_B = 1.38e-23
TEMP_K = 290
B = 10e6

M = 8
Ts = 10000
J = 3
f_c = 1.3   # GHz
N = 20

grid_size = 20.0
min_distance = 2.0
tau = 0.01
velocity_kmph = 40.0

# Jakes / fading
RAYS = 100
K_RICIAN = 4

# LSTM params matching notebook
lag = 10
hidden_dim = 128
num_layers = 1
learning_rate = 1e-3
batch_size = 64
train_frac = 0.95

# Preserve exact notebook dataset logic
PRESERVE_NOTEBOOK_DATA_LOGIC = True
LSTM_SOURCE = "interference"   # notebook used all_Int_pows_dBm

# Plot controls
TRACE_FEATURE_INDEX = 0
TRACE_X_DIVISOR = 10.0         # notebook used xticks = np.arange(len(pred[:,0])) / 10
MAX_TRACE_POINTS = None        # None => use full test trace, like notebook
CI_CONF_LEVEL = 0.95

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


def mean_ci(arr: np.ndarray):
    """
    arr shape: [num_runs, ...]
    returns mean, lower, upper
    """
    arr = np.asarray(arr, dtype=np.float64)
    mean = np.mean(arr, axis=0)

    if arr.shape[0] == 1:
        return mean, mean, mean

    std = np.std(arr, axis=0, ddof=1)
    se = std / np.sqrt(arr.shape[0])
    crit = t_critical_95(arr.shape[0])
    half = crit * se
    return mean, mean - half, mean + half


def compute_gamma0():
    noise_spectral_density = k_B * TEMP_K
    noise_spectral_density_W = noise_spectral_density * 1000.0
    noise_spectral_density_dBm = 10 * np.log10(noise_spectral_density_W)
    n_thermal = noise_spectral_density * B
    n_thermal_dBm = 10 * np.log10(n_thermal * 1000.0)
    gamma_0_dB = Pt_dBm - n_thermal_dBm
    gamma_0 = np.power(10.0, gamma_0_dB / 10.0)
    return {
        "noise_spectral_density_dBm": float(noise_spectral_density_dBm),
        "n_thermal_dBm": float(n_thermal_dBm),
        "gamma_0_dB": float(gamma_0_dB),
        "gamma_0": float(gamma_0),
    }


# ============================================================
# Mobility / geometry functions matching notebook
# ============================================================

def generate_random_point(grid_size):
    return np.random.uniform(0, grid_size, 2)


def generate_random_velocity():
    angle = np.random.uniform(0, 2 * np.pi)
    velocity = velocity_kmph * 5.0 / 18.0
    return np.array([velocity * np.cos(angle), velocity * np.sin(angle)])


def is_within_grid(point, grid_size):
    return all(0 <= coord <= grid_size for coord in point)


def handle_boundary_collisions(point, grid_size):
    angle = np.random.uniform(0, 2 * np.pi)
    velocity = velocity_kmph * 5.0 / 18.0
    if point[0] <= 0 or point[0] >= grid_size:
        angle = np.pi - angle if point[0] <= 0 else -angle
    if point[1] <= 0 or point[1] >= grid_size:
        angle = -np.pi / 2 - angle if point[1] <= 0 else np.pi / 2 - angle
    return np.array([velocity * np.cos(angle), velocity * np.sin(angle)])


def handle_ap_collisions(points, velocities, min_distance):
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < min_distance:
                velocities[i] = generate_random_velocity()
                velocities[j] = generate_random_velocity()
    return velocities


def update_positions(points, velocities, tau, grid_size, min_distance):
    new_points = points + velocities * tau
    for i, point in enumerate(new_points):
        if not is_within_grid(point, grid_size):
            velocities[i] = handle_boundary_collisions(point, grid_size)
        new_points[i] = points[i] + velocities[i] * tau
    velocities = handle_ap_collisions(new_points, velocities, min_distance)
    return new_points, velocities


def return_euclid_dist(device_x_coord, device_y_coord, AP_x_coord, AP_y_coord):
    device_coords = np.array([device_x_coord, device_y_coord])
    AP_coords = np.array([AP_x_coord, AP_y_coord])
    return np.linalg.norm(device_coords - AP_coords)


def simulate_subnetwork_centers():
    points = np.array([generate_random_point(grid_size) for _ in range(M)])
    velocities = np.array([generate_random_velocity() for _ in range(M)])
    sub_net_cents = np.zeros((Ts + 1, M, 2), dtype=np.float64)
    sub_net_cents[0] = points

    for step in range(Ts - 1):
        points, velocities = update_positions(points, velocities, tau, grid_size, min_distance)
        sub_net_cents[step + 1] = points
    return sub_net_cents[:Ts]


def simulate_device_relative_locations():
    d_relative_locs = {}
    for i in range(M):
        d_angle = np.random.uniform(0, 2 * np.pi, J)
        d_r = np.random.uniform(0, 1, J)
        d_relative_locs[i] = np.vstack([d_r * np.cos(d_angle), d_r * np.sin(d_angle)])
    return d_relative_locs


def compute_device_coordinates(sub_net_cents, d_relative_locs):
    x_coords_ts, y_coords_ts = {}, {}
    for ts in range(Ts):
        coords = sub_net_cents[ts]
        point_xs, point_ys = [], []
        for k in d_relative_locs.keys():
            point_x = d_relative_locs[k][0] + coords[k][0]
            point_y = d_relative_locs[k][1] + coords[k][1]
            point_xs.append(point_x)
            point_ys.append(point_y)
        x_coords_ts[ts] = point_xs
        y_coords_ts[ts] = point_ys
    return x_coords_ts, y_coords_ts


def compute_tx_rx_distances(sub_net_cents, x_coords_ts, y_coords_ts):
    TxRxds = np.zeros((Ts, M, M * J), dtype=np.float64)
    for ts in range(Ts):
        device_x_coords = np.array(x_coords_ts[ts]).flatten()
        device_y_coords = np.array(y_coords_ts[ts]).flatten()
        AP_x_coords = sub_net_cents[ts][:, 0]
        AP_y_coords = sub_net_cents[ts][:, 1]

        dists = np.zeros((M, M * J), dtype=np.float64)
        for i in range(AP_x_coords.shape[0]):
            dist = []
            for j_idx in range(len(device_x_coords)):
                dist.append(
                    return_euclid_dist(
                        device_x_coords[j_idx],
                        device_y_coords[j_idx],
                        AP_x_coords[i],
                        AP_y_coords[i],
                    )
                )
            dists[i] = np.array(dist)
        TxRxds[ts] = dists
    return TxRxds


# ============================================================
# Jakes fading and SINR / interference simulation
# ============================================================

def return_jakes_coeffcients(fd_max, TimeVaris, n_links=5, plot=False):
    ff_gains = []
    TimeSequences = []

    for _ in tqdm(range(n_links), desc="Generating Jakes coefficients", leave=False):
        frequs = np.sort(
            np.array(
                [np.round(fd_max * np.cos(2 * np.pi * np.random.uniform(0, 1))) for _ in range(RAYS)]
            )
        )
        phases = np.array([np.exp(1j * 2 * np.pi * np.random.uniform(0, 1)) for _ in range(RAYS)])

        expo = np.exp(1j * 2 * np.pi * np.outer(TimeVaris, frequs))
        time_sequence = expo @ phases

        power_sequence = np.abs(time_sequence) ** 2
        ff_gains.append(power_sequence)
        TimeSequences.append(time_sequence)

    ff_gains = np.array(ff_gains, dtype=np.float64) / RAYS
    TimeSequences = np.array(TimeSequences, dtype=np.complex128)

    if plot and len(ff_gains) > 0:
        plt.figure()
        plt.plot(TimeVaris[:200], 10 * np.log10(ff_gains[0][:200]))
        plt.grid()
        plt.tight_layout()
        plt.show()

    return ff_gains, TimeSequences


def simulate_one_realization_arrays():
    gamma_info = compute_gamma0()
    gamma_0 = gamma_info["gamma_0"]

    sub_net_cents = simulate_subnetwork_centers()
    d_relative_locs = simulate_device_relative_locations()
    x_coords_ts, y_coords_ts = compute_device_coordinates(sub_net_cents, d_relative_locs)
    TxRxds = compute_tx_rx_distances(sub_net_cents, x_coords_ts, y_coords_ts)

    v_ms = velocity_kmph * 5.0 / 18.0
    c = 3e8
    fd_max = v_ms * f_c * 1e9 / c
    TimeVaris = np.arange(0, 5, 5 / Ts)

    ff_gains, _ = return_jakes_coeffcients(
        fd_max, TimeVaris, n_links=M * (M - 1) * J * N, plot=False
    )

    LOS_scale = np.sqrt(K_RICIAN / (K_RICIAN + 1))
    NLOS_scale = np.sqrt(1 / (K_RICIAN + 1))

    random_phase = np.exp(1j * 2 * np.pi * np.random.rand(M, J, N))
    LOS_component = LOS_scale * np.ones((M, J, N)) * random_phase
    NLOS_component = NLOS_scale * (
        np.random.normal(0, 1 / np.sqrt(2), (M, J, N))
        + 1j * np.random.normal(0, 1 / np.sqrt(2), (M, J, N))
    )
    FastFadingChannels = LOS_component + NLOS_component
    FadingGains = np.abs(FastFadingChannels) ** 2

    all_SINRsdB = np.zeros((Ts, M, J, N), dtype=np.float32)
    alltime_WantedSigPerDev = []
    alltime_InterfPowsPerDev = []

    for ts in range(Ts):
        all_fast_fading_gains = np.zeros((M, M * J, N), dtype=np.float64)

        # This matches the notebook logic exactly:
        # jakes_coeffs = ff_gains[:, ts]
        # jakes_coeffs = reshape(M, (M-1)*J, N)
        jakes_coeffs = ff_gains[:, ts]
        jakes_coeffs = np.reshape(jakes_coeffs, (M, (M - 1) * J, N))

        for m_idx in range(M):
            all_fast_fading_gains[m_idx] = np.concatenate(
                [FadingGains[m_idx], jakes_coeffs[m_idx]],
                axis=0
            )

        # Small safety floor; does not affect normal notebook behavior
        txrx_safe = np.maximum(TxRxds[ts], 1e-12)

        PL_los = 31.84 + 21.5 * np.log10(txrx_safe) + 19 * np.log10(f_c)
        PL = 33.0 + 25.5 * np.log10(txrx_safe) + 20 * np.log10(f_c)
        PL_nlos = np.maximum(PL_los, PL)

        PathGains = np.power(10.0, -PL_nlos / 10.0)
        PathGains = np.repeat(PathGains[:, :, np.newaxis], N, axis=2)
        PathGainsTot = PathGains * all_fast_fading_gains

        WantedSigPerDev = np.zeros((M, J, N), dtype=np.float64)
        for m_idx in range(M):
            WantedSigPerDev[m_idx] = PathGainsTot[m_idx, m_idx * J:(m_idx + 1) * J]

        InterfPowsPerDev = np.zeros((M, J, N), dtype=np.float64)
        for m_idx in range(M):
            interferers = [i for i in range(M) if i != m_idx]
            devs = np.arange(m_idx * J, (m_idx + 1) * J)
            interf_pow_gains = PathGainsTot[np.ix_(interferers, devs, np.arange(N))]
            InterfPowsPerDev[m_idx] = np.sum(interf_pow_gains, axis=0)

        alltime_WantedSigPerDev.append(WantedSigPerDev)
        alltime_InterfPowsPerDev.append(InterfPowsPerDev)

        SINRs = WantedSigPerDev / (InterfPowsPerDev + 1.0 / gamma_0)
        SINRsdB = 10.0 * np.log10(np.maximum(SINRs, 1e-30))
        all_SINRsdB[ts] = SINRsdB.astype(np.float32)

    alltime_WantedSigPerDev = np.array(alltime_WantedSigPerDev, dtype=np.float32)
    alltime_InterfPowsPerDev = np.array(alltime_InterfPowsPerDev, dtype=np.float32)
    all_Int_pows_dBm = 10.0 * np.log10(np.maximum(alltime_InterfPowsPerDev, 1e-30)).astype(np.float32)

    return {
        "sub_net_cents": sub_net_cents.astype(np.float32),
        "TxRxds": TxRxds.astype(np.float32),
        "ff_gains": ff_gains.astype(np.float32),
        "all_SINRsdB": all_SINRsdB,
        "alltime_WantedSigPerDev": alltime_WantedSigPerDev,
        "alltime_InterfPowsPerDev": alltime_InterfPowsPerDev,
        "all_Int_pows_dBm": all_Int_pows_dBm,
        "gamma_info": gamma_info,
    }


# ============================================================
# LSTM model matching notebook
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def build_lstm_dataset(all_SINRsdB, all_Int_pows_dBm):
    inp_data, out_data = [], []

    if LSTM_SOURCE.lower() == "sinr":
        source_cube = all_SINRsdB
    else:
        source_cube = all_Int_pows_dBm

    # Preserve exact notebook logic
    # The notebook repeatedly appends the same subnetwork/device data M times.
    # We keep that here to reproduce the same behavior and the same loss-curve scale.
    if PRESERVE_NOTEBOOK_DATA_LOGIC:
        for _ in range(M):
            sinr_sub_nw = source_cube[:, 0, :, :]
            data_per_device = sinr_sub_nw[:, 0, :]
            for t in range(0, len(data_per_device) - lag):
                inp_data.append(data_per_device[t:t + lag])
                out_data.append(data_per_device[t + lag])
    else:
        for m_idx in range(source_cube.shape[1]):
            for j_idx in range(source_cube.shape[2]):
                data_per_device = source_cube[:, m_idx, j_idx, :]
                for t in range(0, len(data_per_device) - lag):
                    inp_data.append(data_per_device[t:t + lag])
                    out_data.append(data_per_device[t + lag])

    inp_data = np.array(inp_data, dtype=np.float32)
    out_data = np.array(out_data, dtype=np.float32)
    return inp_data, out_data


def run_lstm_training(inp_data, out_data, run_seed):
    """
    One-step-ahead prediction only:
    input window  [t, ..., t+lag-1]
    target output [t+lag]
    No recursive multi-step rollout is used.
    """
    set_all_seeds(run_seed)

    samp_no = int(train_frac * len(inp_data))
    inp_train_data = inp_data[:samp_no]
    out_train_data = out_data[:samp_no]
    inp_test_data = inp_data[samp_no:]
    out_test_data = out_data[samp_no:]

    inp_train_t = torch.tensor(inp_train_data, dtype=torch.float32)
    out_train_t = torch.tensor(out_train_data, dtype=torch.float32)
    inp_test_t = torch.tensor(inp_test_data, dtype=torch.float32)
    out_test_t = torch.tensor(out_test_data, dtype=torch.float32)

    train_dataset = TensorDataset(inp_train_t, out_train_t)

    generator = torch.Generator()
    generator.manual_seed(run_seed)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False
    )

    input_dim = N
    output_dim = N

    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # IMPORTANT: record loss per mini-batch update to match notebook plot
    batch_loss_history = []

    for _ in tqdm(range(NUM_EPOCHS), desc=f"Training seed {run_seed}", leave=False):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss_history.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        test_inputs = inp_test_t.to(DEVICE)
        test_targets = out_test_t.to(DEVICE)
        test_outputs = model(test_inputs)

    pred = test_outputs.detach().cpu().numpy().astype(np.float32)
    tgt = test_targets.detach().cpu().numpy().astype(np.float32)

    test_mse = float(np.mean((pred - tgt) ** 2))

    return {
        "train_loss_steps": np.array(batch_loss_history, dtype=np.float32),
        "pred": pred,
        "tgt": tgt,
        "inp_test": inp_test_data.astype(np.float32),
        "num_train_batches_per_epoch": len(train_loader),
        "test_mse": test_mse,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }


# ============================================================
# Plotting
# ============================================================

def plot_training_loss_ci(train_mean, train_lo, train_hi, out_png, out_pdf):
    steps = np.arange(len(train_mean))

    plt.figure(figsize=(6.6, 4.3))
    plt.semilogy(
        steps,
        np.clip(train_mean, 1e-12, None),
        linewidth=1.6,
        label="Training MSE loss"
    )
    plt.fill_between(
        steps,
        np.clip(train_lo, 1e-12, None),
        np.clip(train_hi, 1e-12, None),
        alpha=0.20,
        label="95% CI",
    )
    plt.ylabel("Loss(log scale)")
    plt.xlabel("Mini-batch updates")
    plt.grid(True, which="both", linestyle="-", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def plot_interference_trace_ci_fixed_target(
    tgt_reference,
    pred_runs,
    feature_idx,
    trace_x_divisor,
    max_points,
    out_png,
    out_pdf,
):
    """
    FIX_SIMULATION_ACROSS_RUNS = True:
    - same target across runs
    - CI only on predicted trace across independent trainings
    """
    n_total = tgt_reference.shape[0]
    npts = n_total if max_points is None else min(max_points, n_total)

    xticks = np.arange(npts, dtype=np.float64) / trace_x_divisor
    tgt_line = tgt_reference[:npts, feature_idx]
    pred_sel = pred_runs[:, :npts, feature_idx]

    pred_mean, pred_lo, pred_hi = mean_ci(pred_sel)

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(xticks, tgt_line, linewidth=1.8, label="target")
    plt.plot(xticks, pred_mean, linewidth=1.8, label="predicted mean")
    plt.fill_between(xticks, pred_lo, pred_hi, alpha=0.20, label="predicted 95% CI")
    plt.xlabel("Time Steps")
    plt.ylabel("Interference power (dBm)")
    plt.title("Target vs predicted interference power with 95% CI")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    return {
        "xticks": xticks.astype(np.float32),
        "target": tgt_line.astype(np.float32),
        "pred_mean": pred_mean.astype(np.float32),
        "pred_lower": pred_lo.astype(np.float32),
        "pred_upper": pred_hi.astype(np.float32),
    }


# ============================================================
# Main
# ============================================================

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"FIX_SIMULATION_ACROSS_RUNS = {FIX_SIMULATION_ACROSS_RUNS}")

    train_loss_runs = []
    pred_runs = []
    tgt_runs = []

    gamma_info_last = None
    num_train_batches_per_epoch = None
    best_model_state = None
    best_model_test_mse = np.inf
    
    if FIX_SIMULATION_ACROSS_RUNS:
        # ----------------------------------------------------
        # Same simulation for all runs
        # ----------------------------------------------------
        set_all_seeds(BASE_SEED)
        sim = simulate_one_realization_arrays()
        gamma_info_last = sim["gamma_info"]

        all_SINRsdB = sim["all_SINRsdB"]
        all_Int_pows_dBm = sim["all_Int_pows_dBm"]

        inp_data, out_data = build_lstm_dataset(all_SINRsdB, all_Int_pows_dBm)

        np.savez_compressed(
            out_dir / "fixed_simulation_data.npz",
            all_SINRsdB=all_SINRsdB,
            all_Int_pows_dBm=all_Int_pows_dBm,
            inp_data=inp_data,
            out_data=out_data,
        )

        for run_idx in range(NUM_RUNS):
            run_seed = BASE_SEED + 1000 + run_idx
            print(f"Training run {run_idx + 1}/{NUM_RUNS}, seed={run_seed}")
            lstm_out = run_lstm_training(inp_data, out_data, run_seed)

            train_loss_runs.append(lstm_out["train_loss_steps"])
            pred_runs.append(lstm_out["pred"])
            tgt_runs.append(lstm_out["tgt"])
            num_train_batches_per_epoch = lstm_out["num_train_batches_per_epoch"]

            print(f"  Test MSE for run {run_idx + 1}: {lstm_out['test_mse']:.6f}")

            if SAVE_BEST_MODEL and lstm_out["test_mse"] < best_model_test_mse:
                best_model_test_mse = lstm_out["test_mse"]
                best_model_state = lstm_out["model_state_dict"]


        if best_model_state is not None:
            model_save_path = out_dir / MODEL_SAVE_NAME
            torch.save(best_model_state, model_save_path)
            print(f"\nSaved best LSTM model to: {model_save_path.resolve()}")
            print(f"Best model test MSE: {best_model_test_mse:.6f}")

    else:
        raise NotImplementedError(
            "For notebook-like CI plots, keep FIX_SIMULATION_ACROSS_RUNS = True."
        )

    train_loss_runs = np.stack(train_loss_runs, axis=0).astype(np.float32)   # [R, num_updates]
    pred_runs = np.stack(pred_runs, axis=0).astype(np.float32)               # [R, Ttest, N]
    tgt_runs = np.stack(tgt_runs, axis=0).astype(np.float32)                 # [R, Ttest, N]

    # --------------------------------------------------------
    # Save raw arrays
    # --------------------------------------------------------
    np.save(out_dir / "train_loss_runs.npy", train_loss_runs)
    np.save(out_dir / "pred_test_runs.npy", pred_runs)
    np.save(out_dir / "tgt_test_runs.npy", tgt_runs)

    # --------------------------------------------------------
    # Training loss CI
    # --------------------------------------------------------
    train_mean, train_lo, train_hi = mean_ci(train_loss_runs)

    plot_training_loss_ci(
        train_mean=train_mean,
        train_lo=train_lo,
        train_hi=train_hi,
        out_png=out_dir / "training_mse_loss_ci.png",
        out_pdf=out_dir / "training_mse_loss_ci.pdf",
    )

    # --------------------------------------------------------
    # Interference trace CI
    # --------------------------------------------------------
    trace_bundle = plot_interference_trace_ci_fixed_target(
        tgt_reference=tgt_runs[0],
        pred_runs=pred_runs,
        feature_idx=TRACE_FEATURE_INDEX,
        trace_x_divisor=TRACE_X_DIVISOR,
        max_points=MAX_TRACE_POINTS,
        out_png=out_dir / "interference_trace_ci.png",
        out_pdf=out_dir / "interference_trace_ci.pdf",
    )

    np.savez_compressed(
        out_dir / "processed_plot_data.npz",
        train_steps=np.arange(train_loss_runs.shape[1], dtype=np.int32),
        train_mean=train_mean.astype(np.float32),
        train_lower=train_lo.astype(np.float32),
        train_upper=train_hi.astype(np.float32),
        xticks=trace_bundle["xticks"],
        target=trace_bundle["target"],
        pred_mean=trace_bundle["pred_mean"],
        pred_lower=trace_bundle["pred_lower"],
        pred_upper=trace_bundle["pred_upper"],
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------
    metadata = {
        "num_runs": NUM_RUNS,
        "num_epochs": NUM_EPOCHS,
        "base_seed": BASE_SEED,
        "fix_simulation_across_runs": FIX_SIMULATION_ACROSS_RUNS,
        "one_step_ahead_prediction": True,
        "training_loss_recorded_per": "mini_batch_update",
        "simulation_params": {
            "Pt_dBm": Pt_dBm,
            "B_Hz": B,
            "M": M,
            "Ts": Ts,
            "J": J,
            "f_c_GHz": f_c,
            "N": N,
            "grid_size": grid_size,
            "min_distance": min_distance,
            "tau": tau,
            "velocity_kmph": velocity_kmph,
            "RAYS": RAYS,
            "K_RICIAN": K_RICIAN,
        },
        "lstm_params": {
            "lag": lag,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "train_frac": train_frac,
            "LSTM_SOURCE": LSTM_SOURCE,
            "PRESERVE_NOTEBOOK_DATA_LOGIC": PRESERVE_NOTEBOOK_DATA_LOGIC,
        },
        "trace_feature_index": TRACE_FEATURE_INDEX,
        "trace_x_divisor": TRACE_X_DIVISOR,
        "max_trace_points": MAX_TRACE_POINTS,
        "num_train_batches_per_epoch": num_train_batches_per_epoch,
        "device": str(DEVICE),
        "gamma_info": gamma_info_last,
        "raw_shapes": {
            "train_loss_runs": list(train_loss_runs.shape),
            "pred_test_runs": list(pred_runs.shape),
            "tgt_test_runs": list(tgt_runs.shape),
        },
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved outputs in:", out_dir.resolve())
    for name in [
        "training_mse_loss_ci.png",
        "training_mse_loss_ci.pdf",
        "interference_trace_ci.png",
        "interference_trace_ci.pdf",
        "train_loss_runs.npy",
        "pred_test_runs.npy",
        "tgt_test_runs.npy",
        "processed_plot_data.npz",
        "metadata.json",
    ]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()