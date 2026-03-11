#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Corrected Monte-Carlo SINR CDF runner with confidence intervals.

What this script does
---------------------
1) Reproduces the same three SINR-related CDFs as in the notebook:
   - F(gamma)
   - F(gamma_bar)
   - F(gamma - gamma_bar | gamma_bar)

2) Runs the whole simulation multiple times (default: 5 runs)

3) Plots:
   - main line = exact pooled empirical CDF, same style as the notebook
   - confidence band = pointwise t-based CI across Monte-Carlo runs

4) Saves:
   - PNG and PDF figure
   - processed NPZ for easy replotting later
   - raw per-run .npy files
   - metadata JSON

Usage
-----
python sinr_cdf_ci_runner_fixed.py
python sinr_cdf_ci_runner_fixed.py --num-runs 5 --output-dir sinr_cdf_ci_outputs_fixed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import t as student_t
except Exception:
    student_t = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# ============================================================
# Utility functions
# ============================================================

def return_cdf(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exact empirical CDF in the same style as the notebook."""
    a = np.asarray(a).reshape(-1)
    sorted_a = np.sort(a)
    cdf = np.arange(1, len(sorted_a) + 1, dtype=np.float64) / float(len(sorted_a))
    return sorted_a, cdf


def empirical_cdf_on_grid(samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Evaluate ECDF on a common grid, used only for CI computation."""
    samples = np.asarray(samples).reshape(-1)
    sorted_samples = np.sort(samples)
    counts = np.searchsorted(sorted_samples, x_grid, side="right")
    return counts.astype(np.float64) / float(sorted_samples.size)


def t_critical(conf_level: float, df: int) -> float:
    if df <= 0:
        return 0.0
    if student_t is not None:
        return float(student_t.ppf(0.5 + conf_level / 2.0, df))
    if abs(conf_level - 0.95) < 1e-12 and df == 4:
        return 2.7764451051977987
    return 1.96


def return_noise_and_gamma0(pt_dbm: float, bandwidth_hz: float) -> Tuple[float, float]:
    k_b = 1.38e-23
    temp_k = 290.0
    n_thermal = k_b * temp_k * bandwidth_hz
    n_thermal_dbm = 10.0 * np.log10(n_thermal * 1000.0)
    gamma_0_db = pt_dbm - n_thermal_dbm
    gamma_0 = 10.0 ** (gamma_0_db / 10.0)
    return float(n_thermal_dbm), float(gamma_0)


# ============================================================
# Mobility / geometry
# ============================================================

def generate_random_velocity(rng: np.random.Generator, velocity_ms: float) -> np.ndarray:
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return np.array([velocity_ms * np.cos(angle), velocity_ms * np.sin(angle)], dtype=np.float64)


def handle_boundary_collisions(
    point: np.ndarray,
    grid_size: float,
    velocity_ms: float,
    rng: np.random.Generator,
) -> np.ndarray:
    angle = rng.uniform(0.0, 2.0 * np.pi)
    if point[0] <= 0.0 or point[0] >= grid_size:
        angle = np.pi - angle if point[0] <= 0.0 else -angle
    if point[1] <= 0.0 or point[1] >= grid_size:
        angle = -np.pi / 2.0 - angle if point[1] <= 0.0 else np.pi / 2.0 - angle
    return np.array([velocity_ms * np.cos(angle), velocity_ms * np.sin(angle)], dtype=np.float64)


def handle_ap_collisions(
    points: np.ndarray,
    velocities: np.ndarray,
    min_distance: float,
    velocity_ms: float,
    rng: np.random.Generator,
) -> np.ndarray:
    num_points = len(points)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            if np.linalg.norm(points[i] - points[j]) < min_distance:
                velocities[i] = generate_random_velocity(rng, velocity_ms)
                velocities[j] = generate_random_velocity(rng, velocity_ms)
    return velocities


def simulate_ap_centers(
    ts: int,
    m: int,
    grid_size: float,
    min_distance: float,
    tau: float,
    velocity_ms: float,
    rng: np.random.Generator,
) -> np.ndarray:
    points = rng.uniform(0.0, grid_size, size=(m, 2))
    velocities = np.stack([generate_random_velocity(rng, velocity_ms) for _ in range(m)], axis=0)

    centers = np.zeros((ts, m, 2), dtype=np.float64)
    centers[0] = points

    for step in range(1, ts):
        new_points = points + velocities * tau

        for i in range(m):
            if not ((0.0 <= new_points[i][0] <= grid_size) and (0.0 <= new_points[i][1] <= grid_size)):
                velocities[i] = handle_boundary_collisions(new_points[i], grid_size, velocity_ms, rng)

        new_points = points + velocities * tau
        velocities = handle_ap_collisions(new_points, velocities, min_distance, velocity_ms, rng)

        points = new_points
        centers[step] = points

    return centers


def build_distances(
    centers: np.ndarray,
    j: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Same style as the notebook:
    device offsets are fixed relative to their serving AP over time.
    """
    ts, m, _ = centers.shape

    d_angle = rng.uniform(0.0, 2.0 * np.pi, size=(m, j))
    d_r = rng.uniform(0.0, 1.0, size=(m, j))

    rel_x = d_r * np.cos(d_angle)
    rel_y = d_r * np.sin(d_angle)

    tx_rx_ds = np.zeros((ts, m, m * j), dtype=np.float64)

    for t_idx in range(ts):
        coords = centers[t_idx]
        device_x = rel_x + coords[:, 0:1]
        device_y = rel_y + coords[:, 1:2]
        device_coords = np.stack([device_x, device_y], axis=-1).reshape(m * j, 2)
        ap_coords = coords
        dists = np.linalg.norm(ap_coords[:, None, :] - device_coords[None, :, :], axis=2)
        tx_rx_ds[t_idx] = dists

    return tx_rx_ds


# ============================================================
# Fading
# ============================================================

def return_jakes_coefficients(
    fd_max: float,
    time_values: np.ndarray,
    n_links: int,
    rng: np.random.Generator,
    rays: int = 100,
) -> np.ndarray:
    """
    Same idea as notebook:
    ff_gains shape = (n_links, Ts)
    """
    ff_gains = np.zeros((n_links, len(time_values)), dtype=np.float64)

    for i in tqdm(range(n_links), desc="Generating Jakes coefficients", leave=False):
        frequs = np.sort(
            np.array([np.round(fd_max * np.cos(2 * np.pi * rng.uniform(0, 1))) for _ in range(rays)], dtype=np.float64)
        )
        phases = np.array([np.exp(1j * 2 * np.pi * rng.uniform(0, 1)) for _ in range(rays)], dtype=np.complex128)

        time_sequence = np.zeros(len(time_values), dtype=np.complex128)
        for t_idx, t in enumerate(time_values):
            tab = np.exp(1j * 2 * np.pi * frequs * t)
            tabrot = tab * phases
            time_sequence[t_idx] = np.sum(tabrot)

        power_sequence = np.abs(time_sequence) ** 2
        ff_gains[i] = power_sequence / rays

    return ff_gains


def build_all_fast_fading_gains(
    fading_gains_desired: np.ndarray,
    ff_gains_ts: np.ndarray,
    m: int,
    j: int,
    n: int,
) -> np.ndarray:
    """
    Same placement logic as the notebook.
    """
    jakes_coeffs = np.reshape(ff_gains_ts, (m, (m - 1) * j, n))
    all_fast_fading_gains = np.zeros((m, m * j, n), dtype=np.float64)

    for subnet_idx in range(m):
        all_fast_fading_gains[subnet_idx, subnet_idx * j:(subnet_idx + 1) * j, :] = fading_gains_desired[subnet_idx]

        ptr = 0
        for other_idx in range(m):
            if other_idx == subnet_idx:
                continue
            all_fast_fading_gains[subnet_idx, other_idx * j:(other_idx + 1) * j, :] = jakes_coeffs[subnet_idx, ptr:ptr + j, :]
            ptr += j

    return all_fast_fading_gains


# ============================================================
# One simulation run
# ============================================================

def simulate_once(params: Dict[str, float | int], seed: int, quiet: bool = False) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    m = int(params["M"])
    ts = int(params["Ts"])
    j = int(params["J"])
    f_c = float(params["f_c_GHz"])
    n = int(params["N"])
    pt_dbm = float(params["Pt_dBm"])
    bandwidth_hz = float(params["B_Hz"])
    grid_size = float(params["grid_size_m"])
    min_distance = float(params["min_distance_m"])
    tau = float(params["tau_s"])
    velocity_kmph = float(params["velocity_kmph"])
    rician_k = float(params["rician_K"])
    eps = float(params["eps"])
    jakes_rays = int(params["jakes_rays"])
    time_horizon_s = float(params["time_horizon_s"])

    _, gamma_0 = return_noise_and_gamma0(pt_dbm, bandwidth_hz)

    velocity_ms = velocity_kmph * 5.0 / 18.0

    centers = simulate_ap_centers(
        ts=ts,
        m=m,
        grid_size=grid_size,
        min_distance=min_distance,
        tau=tau,
        velocity_ms=velocity_ms,
        rng=rng,
    )

    tx_rx_ds = build_distances(centers=centers, j=j, rng=rng)

    c = 3e8
    fd_max = velocity_ms * f_c * 1e9 / c
    time_values = np.linspace(0.0, time_horizon_s, ts, endpoint=False, dtype=np.float64)

    n_links = m * (m - 1) * j * n
    ff_gains = return_jakes_coefficients(
        fd_max=fd_max,
        time_values=time_values,
        n_links=n_links,
        rng=rng,
        rays=jakes_rays,
    )

    # Rician desired links only
    los_scale = np.sqrt(rician_k / (rician_k + 1.0))
    nlos_scale = np.sqrt(1.0 / (rician_k + 1.0))

    random_phase = np.exp(1j * 2 * np.pi * rng.random((m, j, n)))
    los_component = los_scale * np.ones((m, j, n), dtype=np.float64) * random_phase
    nlos_component = nlos_scale * (
        rng.normal(0, 1 / np.sqrt(2), (m, j, n)) +
        1j * rng.normal(0, 1 / np.sqrt(2), (m, j, n))
    )

    fast_fading_channels_desired = los_component + nlos_component
    fading_gains_desired = np.abs(fast_fading_channels_desired) ** 2

    all_sinr_db = np.zeros((ts, m, j, n), dtype=np.float64)
    all_means_per_subnw = np.zeros((ts, m), dtype=np.float64)
    all_diffs_from_mean = np.zeros((ts, m, j, n), dtype=np.float64)

    iterator = range(ts)
    if not quiet:
        iterator = tqdm(iterator, desc=f"Sim seed {seed}", leave=False)

    for t_idx in iterator:
        all_fast_fading_gains = build_all_fast_fading_gains(
            fading_gains_desired=fading_gains_desired,
            ff_gains_ts=ff_gains[:, t_idx],
            m=m,
            j=j,
            n=n,
        )

        tx_rxd_safe = np.maximum(tx_rx_ds[t_idx], 1e-3)

        pl_los = 31.84 + 21.5 * np.log10(tx_rxd_safe) + 19.0 * np.log10(f_c)
        pl_alt = 33.0 + 25.5 * np.log10(tx_rxd_safe) + 20.0 * np.log10(f_c)
        pl_nlos = np.maximum(pl_los, pl_alt)

        path_gains = np.power(10.0, -pl_nlos / 10.0)
        path_gains = np.repeat(path_gains[:, :, np.newaxis], n, axis=2)

        path_gains_tot = path_gains * all_fast_fading_gains

        wanted_sig_per_dev = np.zeros((m, j, n), dtype=np.float64)
        interf_pows_per_dev = np.zeros((m, j, n), dtype=np.float64)

        for subnet_idx in range(m):
            wanted_sig_per_dev[subnet_idx] = path_gains_tot[subnet_idx, subnet_idx * j:(subnet_idx + 1) * j, :]

        for subnet_idx in range(m):
            interferers = [idx for idx in range(m) if idx != subnet_idx]
            devs = np.arange(subnet_idx * j, (subnet_idx + 1) * j)
            interf_pow_gains = path_gains_tot[np.ix_(interferers, devs, np.arange(n))]
            interf_pows_per_dev[subnet_idx] = np.sum(interf_pow_gains, axis=0)

        sinrs = wanted_sig_per_dev / (interf_pows_per_dev + 1.0 / gamma_0)
        sinr_db = 10.0 * np.log10(np.maximum(sinrs, eps))

        means_per_subnet = np.mean(sinr_db, axis=(1, 2))
        diffs_from_mean = sinr_db - means_per_subnet[:, np.newaxis, np.newaxis]

        all_sinr_db[t_idx] = sinr_db
        all_means_per_subnw[t_idx] = means_per_subnet
        all_diffs_from_mean[t_idx] = diffs_from_mean

    sinr_flat = all_sinr_db.reshape(-1).astype(np.float32)
    mean_flat = all_means_per_subnw.reshape(-1).astype(np.float32)
    diff_flat = all_diffs_from_mean.reshape(-1).astype(np.float32)

    # Safety checks to catch the exact issue that happened before
    if np.all(sinr_flat == 0):
        raise RuntimeError(f"Run with seed={seed} produced all-zero SINR samples.")
    if np.all(mean_flat == 0):
        raise RuntimeError(f"Run with seed={seed} produced all-zero subnet-mean samples.")
    if np.all(diff_flat == 0):
        raise RuntimeError(f"Run with seed={seed} produced all-zero diff samples.")

    return {
        "sinr_db_samples": sinr_flat,
        "mean_samples": mean_flat,
        "diff_samples": diff_flat,
    }


# ============================================================
# CI bundle
# ============================================================

def build_ci_bundle(run_samples: np.ndarray, conf_level: float, grid_points: int) -> Dict[str, np.ndarray]:
    """
    run_samples shape = (num_runs, num_samples)

    For plotting:
    - pooled_line_x / pooled_line_cdf -> exact empirical CDF of pooled samples
    - x_grid / lower / upper -> CI band
    """
    num_runs = int(run_samples.shape[0])

    pooled = run_samples.reshape(-1)
    pooled_line_x, pooled_line_cdf = return_cdf(pooled)

    x_min = float(np.min(pooled))
    x_max = float(np.max(pooled))
    if np.isclose(x_min, x_max):
        x_grid = np.array([x_min], dtype=np.float64)
    else:
        x_grid = np.linspace(x_min, x_max, int(grid_points), dtype=np.float64)

    cdfs = np.zeros((num_runs, x_grid.size), dtype=np.float64)
    for i in range(num_runs):
        cdfs[i] = empirical_cdf_on_grid(run_samples[i], x_grid)

    mean_cdf = np.mean(cdfs, axis=0)

    if num_runs > 1:
        se = np.std(cdfs, axis=0, ddof=1) / np.sqrt(num_runs)
        crit = t_critical(conf_level, num_runs - 1)
        half_width = crit * se
    else:
        half_width = np.zeros_like(mean_cdf)

    lower = np.clip(mean_cdf - half_width, 1e-6, 1.0)
    upper = np.clip(mean_cdf + half_width, 1e-6, 1.0)

    return {
        "pooled_line_x": pooled_line_x.astype(np.float32),
        "pooled_line_cdf": pooled_line_cdf.astype(np.float32),
        "x_grid": x_grid.astype(np.float32),
        "mean_cdf": mean_cdf.astype(np.float32),
        "lower_cdf": lower.astype(np.float32),
        "upper_cdf": upper.astype(np.float32),
        "all_cdfs": cdfs.astype(np.float32),
    }


# ============================================================
# Plotting / saving
# ============================================================

def plot_ci_curves(
    bundles: Dict[str, Dict[str, np.ndarray]],
    out_png: Path,
    out_pdf: Path,
    title: str = "SINR CDF with confidence intervals",
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    labels = {
        "sinr": r"$F(\gamma)$",
        "mean": r"$F(\bar{\gamma})$",
        "diff": r"$F(\gamma-\bar{\gamma}\mid \bar{\gamma})$",
    }

    for key in ["sinr", "mean", "diff"]:
        bundle = bundles[key]

        line, = ax.semilogy(
            bundle["pooled_line_x"],
            bundle["pooled_line_cdf"],
            linewidth=1.8,
            label=labels[key],
        )

        ax.fill_between(
            bundle["x_grid"],
            bundle["lower_cdf"],
            bundle["upper_cdf"],
            alpha=0.18,
            color=line.get_color(),
        )

    ax.set_xlabel("SINR (dB)")
    ax.set_ylabel("CDF (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.set_ylim(1e-6, 1.0)
    ax.legend()
    fig.tight_layout()

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def save_processed_npz(out_path: Path, bundles: Dict[str, Dict[str, np.ndarray]], metadata: Dict[str, object]) -> None:
    np.savez(
        out_path,
        metadata_json=np.array(json.dumps(metadata, indent=2)),
        sinr_pooled_x=bundles["sinr"]["pooled_line_x"],
        sinr_pooled_cdf=bundles["sinr"]["pooled_line_cdf"],
        sinr_x=bundles["sinr"]["x_grid"],
        sinr_mean=bundles["sinr"]["mean_cdf"],
        sinr_lower=bundles["sinr"]["lower_cdf"],
        sinr_upper=bundles["sinr"]["upper_cdf"],
        mean_pooled_x=bundles["mean"]["pooled_line_x"],
        mean_pooled_cdf=bundles["mean"]["pooled_line_cdf"],
        mean_x=bundles["mean"]["x_grid"],
        mean_mean=bundles["mean"]["mean_cdf"],
        mean_lower=bundles["mean"]["lower_cdf"],
        mean_upper=bundles["mean"]["upper_cdf"],
        diff_pooled_x=bundles["diff"]["pooled_line_x"],
        diff_pooled_cdf=bundles["diff"]["pooled_line_cdf"],
        diff_x=bundles["diff"]["x_grid"],
        diff_mean=bundles["diff"]["mean_cdf"],
        diff_lower=bundles["diff"]["lower_cdf"],
        diff_upper=bundles["diff"]["upper_cdf"],
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corrected SINR CDF Monte-Carlo runner with confidence intervals.")

    parser.add_argument("--output-dir", type=str, default="sinr_cdf_ci_outputs_fixed")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--conf-level", type=float, default=0.95)
    parser.add_argument("--grid-points", type=int, default=2000)
    parser.add_argument("--quiet", action="store_true")

    # defaults matched to your notebook
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--Ts", type=int, default=10000)
    parser.add_argument("--J", type=int, default=3)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--f-c-GHz", dest="f_c_GHz", type=float, default=1.3)
    parser.add_argument("--Pt-dBm", dest="Pt_dBm", type=float, default=10.0)
    parser.add_argument("--B-Hz", dest="B_Hz", type=float, default=10e6)
    parser.add_argument("--grid-size-m", dest="grid_size_m", type=float, default=20.0)
    parser.add_argument("--min-distance-m", dest="min_distance_m", type=float, default=2.0)
    parser.add_argument("--tau-s", dest="tau_s", type=float, default=0.01)
    parser.add_argument("--velocity-kmph", dest="velocity_kmph", type=float, default=40.0)
    parser.add_argument("--rician-K", dest="rician_K", type=float, default=4.0)
    parser.add_argument("--eps", type=float, default=1e-30)
    parser.add_argument("--jakes-rays", type=int, default=100)
    parser.add_argument("--time-horizon-s", dest="time_horizon_s", type=float, default=5.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "M": args.M,
        "Ts": args.Ts,
        "J": args.J,
        "N": args.N,
        "f_c_GHz": args.f_c_GHz,
        "Pt_dBm": args.Pt_dBm,
        "B_Hz": args.B_Hz,
        "grid_size_m": args.grid_size_m,
        "min_distance_m": args.min_distance_m,
        "tau_s": args.tau_s,
        "velocity_kmph": args.velocity_kmph,
        "rician_K": args.rician_K,
        "eps": args.eps,
        "jakes_rays": args.jakes_rays,
        "time_horizon_s": args.time_horizon_s,
    }

    sinr_runs_list = []
    mean_runs_list = []
    diff_runs_list = []

    run_iter = range(args.num_runs)
    if not args.quiet:
        run_iter = tqdm(run_iter, desc="Monte-Carlo runs")

    for run_idx in run_iter:
        seed = args.base_seed + run_idx
        result = simulate_once(params=params, seed=seed, quiet=args.quiet)

        sinr_runs_list.append(result["sinr_db_samples"])
        mean_runs_list.append(result["mean_samples"])
        diff_runs_list.append(result["diff_samples"])

    sinr_runs = np.stack(sinr_runs_list, axis=0).astype(np.float32)
    mean_runs = np.stack(mean_runs_list, axis=0).astype(np.float32)
    diff_runs = np.stack(diff_runs_list, axis=0).astype(np.float32)

    # final safety checks
    for i in range(args.num_runs):
        if np.all(sinr_runs[i] == 0):
            raise RuntimeError(f"Run {i} in sinr_runs is all zeros.")
        if np.all(mean_runs[i] == 0):
            raise RuntimeError(f"Run {i} in mean_runs is all zeros.")
        if np.all(diff_runs[i] == 0):
            raise RuntimeError(f"Run {i} in diff_runs is all zeros.")

    bundles = {
        "sinr": build_ci_bundle(sinr_runs, conf_level=args.conf_level, grid_points=args.grid_points),
        "mean": build_ci_bundle(mean_runs, conf_level=args.conf_level, grid_points=args.grid_points),
        "diff": build_ci_bundle(diff_runs, conf_level=args.conf_level, grid_points=args.grid_points),
    }

    metadata = {
        "num_runs": args.num_runs,
        "base_seed": args.base_seed,
        "conf_level": args.conf_level,
        "grid_points": args.grid_points,
        "simulation_params": params,
        "raw_shapes": {
            "sinr_runs": list(sinr_runs.shape),
            "mean_runs": list(mean_runs.shape),
            "diff_runs": list(diff_runs.shape),
        },
        "curve_labels": {
            "sinr": r"$F(\gamma)$",
            "mean": r"$F(\bar{\gamma})$",
            "diff": r"$F(\gamma-\bar{\gamma}\mid \bar{\gamma})$",
        },
    }

    # save raw arrays
    np.save(output_dir / "sinr_db_samples.npy", sinr_runs)
    np.save(output_dir / "mean_samples.npy", mean_runs)
    np.save(output_dir / "diff_samples.npy", diff_runs)

    # save processed bundle
    save_processed_npz(output_dir / "sinr_cdf_ci_processed.npz", bundles, metadata)

    # save metadata
    with open(output_dir / "sinr_cdf_ci_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # save figure
    plot_ci_curves(
        bundles=bundles,
        out_png=output_dir / "sinr_cdf_ci.png",
        out_pdf=output_dir / "sinr_cdf_ci.pdf",
        title="SINR CDF with confidence intervals",
    )

    print(f"Saved outputs to: {output_dir.resolve()}")
    print(f"Figure PNG: {output_dir / 'sinr_cdf_ci.png'}")
    print(f"Figure PDF: {output_dir / 'sinr_cdf_ci.pdf'}")
    print(f"Processed NPZ: {output_dir / 'sinr_cdf_ci_processed.npz'}")
    print(f"Raw NPY: {output_dir / 'sinr_db_samples.npy'}")
    print(f"Raw NPY: {output_dir / 'mean_samples.npy'}")
    print(f"Raw NPY: {output_dir / 'diff_samples.npy'}")


if __name__ == "__main__":
    main()