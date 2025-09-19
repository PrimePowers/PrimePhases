#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import textwrap
from dataclasses import dataclass
from typing import Tuple, List
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# ---------- First 40 imaginary parts t_n of the non-trivial Zeta zeros ----------
ZETA_ZEROS = np.array([
    14.134725141, 21.022039639, 25.010857580, 30.424876126, 32.935061588,
    37.586178159, 40.918719012, 43.327073281, 48.005150881, 49.773832478,
    52.970321478, 56.446247697, 59.347044003, 60.831778525, 65.112544048,
    67.079810529, 69.546401711, 72.067157674, 75.704690699, 77.144840069,
    79.337375020, 82.910380854, 84.735492981, 87.425274613, 88.809111208,
    92.491899271, 94.651344041, 95.870634228, 98.831194218, 101.317851006,
    103.725538040, 105.446623052, 107.168611184, 111.029535543, 111.874659177,
    114.320220915, 116.226680321, 118.790782866, 121.370125002, 122.946829294
], dtype=float)

# -------------------------------------------------------------------------------------

def read_results_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads the results from a CSV file. It expects specific column names (with aliases).
    
    Expected columns (with aliases):
      - X:        ["X", "x"]
      - loglogX:  ["loglogx", "loglogX", "llx", "log_log_x"] (if missing: log(log(X)) is calculated)
      - gamma_std:["g_std", "gamma_std", "gamma", "gamma_standard"]
    Returns: X, loglogX, gamma_std (as numpy arrays, sorted in ascending order of X)
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    def pick(aliases: List[str]) -> str:
        for a in aliases:
            if a.lower() in cols:
                return cols[a.lower()]
        return ""

    col_X = pick(["X", "x"])
    if not col_X:
        raise ValueError(f"CSV {path} does not have a column for X (Aliases: X, x)")

    col_ll = pick(["loglogx", "loglogX", "llx", "log_log_x"])
    col_g  = pick(["g_std", "gamma_std", "gamma", "gamma_standard"])

    X = df[col_X].to_numpy(dtype=float)

    if col_ll:
        loglogX = df[col_ll].to_numpy(dtype=float)
    else:
        if np.any(X <= 1.0):
            raise ValueError("loglogX is missing and X contains values <= 1; cannot calculate log log X.")
        loglogX = np.log(np.log(X))

    if col_g:
        g_std = df[col_g].to_numpy(dtype=float)
    else:
        raise ValueError(f"CSV {path} does not have a column for g_std (Aliases: g_std, gamma_std, gamma)")

    # Sort by X (for safety)
    idx = np.argsort(X)
    return X[idx], loglogX[idx], g_std[idx]


@dataclass
class FitResult:
    a: np.ndarray          # cos coefficients (N,)
    b: np.ndarray          # sin coefficients (N,)
    c0: float              # constant term
    c1: float              # linear trend in u
    omegas: np.ndarray     # frequencies used (t_n * omega_scale)
    var_std: float         # Variance(gamma_std) on the fitted tail segment
    var_opt: float         # Variance(gamma_opt_multi) on the same segment
    reduction: float       # var_std / var_opt


def build_design_matrix(u: np.ndarray, omegas: np.ndarray) -> np.ndarray:
    """
    Builds the design matrix for Least Squares fitting:
    [cos(w1 u), ..., cos(wN u), sin(w1 u), ..., sin(wN u), 1, u]
    """
    N = len(omegas)
    C = np.cos(np.outer(u, omegas))  # (m, N)
    S = np.sin(np.outer(u, omegas))  # (m, N)
    ones = np.ones((len(u), 1))
    U = u.reshape(-1, 1)
    Phi = np.hstack([C, S, ones, U])
    return Phi


def least_squares_fit(u: np.ndarray, y: np.ndarray, omegas: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Fits y(u) ~ Sum a_n cos(w_n u) + b_n sin(w_n u) + c0 + c1 * u
    """
    Phi = build_design_matrix(u, omegas)
    coeffs, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    N = len(omegas)
    a = coeffs[:N]
    b = coeffs[N:2*N]
    c0 = coeffs[-2]
    c1 = coeffs[-1]
    return a, b, c0, c1


def eval_model(u: np.ndarray, a: np.ndarray, b: np.ndarray, c0: float, c1: float, omegas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (trend, oscillation); trend = c0 + c1*u, oscillation = Sum a_n cos(w_n u)+b_n sin(w_n u)
    """
    C = np.cos(np.outer(u, omegas))
    S = np.sin(np.outer(u, omegas))
    osc = C @ a + S @ b
    trend = c0 + c1 * u
    return trend, osc


def spectrum_plot(u: np.ndarray, y: np.ndarray, omegas_ref: np.ndarray, out_png: str):
    """
    Performs a simple spectral analysis (FFT) of y(u) (u is approximately equidistant for log-spaced X).
    """
    # If u is not exactly equidistant, we use the average spacing as an approximation.
    du = np.mean(np.diff(u))
    Y = rfft(y - np.mean(y))
    freqs = rfftfreq(len(y), d=du)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, np.abs(Y))
    for w in omegas_ref:
        plt.axvline(w, ls='--', alpha=0.25)
    plt.xlabel(r'$\omega$ (rad per $\log X$)')
    plt.ylabel('FFT magnitude')
    plt.title(r'Spectrum of $y=\gamma_{\rm std}-\log\log X$')
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def trajectory_plot(X: np.ndarray, g_std: np.ndarray, g_opt: np.ndarray, out_png: str):
    plt.figure(figsize=(10, 5))
    plt.plot(X, g_std, label='gamma_std')
    plt.plot(X, g_opt,  label='gamma_opt_multi (first N zeta zeros)')
    plt.xscale('log')
    plt.xlabel('X')
    plt.ylabel('estimator')
    plt.title('Gamma estimators')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def residual_plot(X: np.ndarray, resid: np.ndarray, out_png: str):
    plt.figure(figsize=(11, 4.2))
    plt.plot(X, resid, label='residual (y - fitted osc)')
    plt.xscale('log')
    plt.xlabel('X')
    plt.ylabel('residual')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def run_fit(X: np.ndarray,
            loglogX: np.ndarray,
            g_std: np.ndarray,
            n_modes: int,
            tail: float,
            omega_scale: float) -> Tuple[FitResult, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs the multi-mode fit on the tail and returns the FitResult + diagnostic series.
    """
    # Target variable for spectrum & fit
    y = g_std - loglogX
    u = np.log(X)

    # Tail window
    m = len(X)
    k0 = int(np.floor((1.0 - tail) * m))
    k0 = max(0, min(k0, m-3))
    u_fit = u[k0:]
    y_fit = y[k0:]
    X_fit = X[k0:]
    g_fit = g_std[k0:]

    # Frequencies: unscaled t_n (but optional --omega-scale is available)
    omegas = omega_scale * ZETA_ZEROS[:n_modes].copy()

    # Least Squares
    a, b, c0, c1 = least_squares_fit(u_fit, y_fit, omegas)

    # Prediction (only oscillation is later subtracted from the estimator)
    trend, osc = eval_model(u_fit, a, b, c0, c1, omegas)
    resid = y_fit - (trend + osc)

    # Optimized estimator = gamma_std - osc (DO NOT subtract the trend)
    g_opt = g_fit - osc

    var_std = float(np.var(g_fit, ddof=1)) if len(g_fit) > 1 else 0.0
    var_opt = float(np.var(g_opt, ddof=1)) if len(g_opt) > 1 else 0.0
    reduction = (var_std / var_opt) if var_opt > 0 else np.inf

    fitres = FitResult(a=a, b=b, c0=float(c0), c1=float(c1),
                       omegas=omegas, var_std=var_std, var_opt=var_opt,
                       reduction=reduction)
    return fitres, X_fit, g_fit, g_opt, resid


def write_fit_report(path: str, fit: FitResult):
    lines = []
    lines.append("# Multimode fit report (exact zeta zeros)")
    lines.append("")
    lines.append(f"modes N         : {len(fit.omegas)}")
    lines.append(f"var(gamma_std)  : {fit.var_std:.9g}")
    lines.append(f"var(gamma_opt)  : {fit.var_opt:.9g}")
    lines.append(f"reduction (x)   : {fit.reduction:.5f}")
    lines.append("")
    lines.append(f"trend c0        : {fit.c0:.12g}")
    lines.append(f"trend c1 (u)    : {fit.c1:.12g}")
    lines.append("")
    lines.append("coefficients (omega, a_cos, b_sin):")
    for w, ai, bi in zip(fit.omegas, fit.a, fit.b):
        lines.append(f"  {w:.9f}  {ai:.12g}  {bi:.12g}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser(
        description="Multimode fit of gamma_std using exact Riemann zero frequencies t_n (no scaling).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("csv", help="Input CSV (results_1e9.csv, etc.)")
    p.add_argument("-N", "--n-modes", type=int, default=10, help="Number of Zeta zeros (t_n) to include in the fit")
    p.add_argument("--tail", type=float, default=0.75, help="Last fraction of the data to fit (0..1)")
    p.add_argument("--omega-scale", type=float, default=1.0, help="Frequency scaling (should remain 1.0)")
    p.add_argument("-o", "--out-prefix", default="gopt",
                   help="Prefix for output files (PNG + TXT)")
    args = p.parse_args()

    X, loglogX, g_std = read_results_csv(args.csv)

    fit, X_fit, g_fit, g_opt, resid = run_fit(
        X, loglogX, g_std,
        n_modes=args.n_modes,
        tail=args.tail,
        omega_scale=args.omega_scale
    )

    # --- Plots ---
    trajectory_plot(X_fit, g_fit, g_opt, f"{args.out_prefix}_traj.png")
    spectrum_plot(np.log(X_fit), g_fit - np.log(np.log(X_fit)),
                  fit.omegas, f"{args.out_prefix}_spectrum.png")
    residual_plot(X_fit, resid, f"{args.out_prefix}_resid.png")

    # --- Report ---
    write_fit_report(f"{args.out_prefix}_fit.txt", fit)

    # Console output
    msg = textwrap.dedent(f"""
    Done.
      CSV           : {args.csv}
      modes (N)     : {args.n_modes}
      tail          : {args.tail}
      omega_scale   : {args.omega_scale}
      var(std)      : {fit.var_std:.9g}
      var(opt)      : {fit.var_opt:.9g}
      reduction (x) : {fit.reduction:.5f}
      out           : {args.out_prefix}_traj.png / _spectrum.png / _resid.png / _fit.txt
    """).strip()
    print(msg)


if __name__ == "__main__":
    main()