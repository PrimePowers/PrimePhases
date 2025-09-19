#!/usr/bin/env python3
# phase_stream_runner.py — streaming prime phase analysis up to 1e12
#
# This script implements the core numerical pipeline for the "Prime Phases"
# research project, enabling a streaming analysis of prime numbers up to
# very large bounds (X >= 10^12) without storing a full list of primes.
#
# Key functionalities:
# - **Prime Generation:** Uses a segmented sieve for memory-efficient prime
#   number generation in chunks.
# - **Incremental Sums:** Computes key sums V(X) and S_gamma(X) incrementally
#   using numerically stable Kahan summation to avoid floating-point errors.
# - **Phase Analysis:** Builds a circular histogram of prime phases and
#   performs a circular Kernel Density Estimation (KDE) via Fast Fourier
#   Transform (FFT) to identify "hotspots" and their significance.
# - **Euler-Mascheroni Estimator:** Calculates the standard and a novel
#   phase-optimized Mertens-type estimator for the Euler-Mascheroni constant (gamma),
#   demonstrating variance reduction.
# - **State Management:** Supports checkpointing and resuming to handle
#   long-running computations.
# - **Output Generation:** Writes results to a CSV file and generates plots
#   for theoretical and empirical verification.

import argparse, json, math, time
from pathlib import Path
import numpy as np

# ----------------------------- Utils: Helper Functions --------------------------------

def kahan_add(sum_, c_, x):
    """
    Kahan summation algorithm for highly precise floating-point addition.
    This is crucial for maintaining accuracy on long-running incremental sums.
    
    Args:
        sum_: The current running sum.
        c_: The current compensation term.
        x: The value to add.
    
    Returns:
        A tuple of (new_sum, new_compensation_term).
    """
    y = x - c_
    t = sum_ + y
    c_new = (t - sum_) - y
    return t, c_new

def wrap_2pi(theta):
    """
    Wraps an angle to the range [0, 2pi).
    This is used for the circular phase space of primes.
    """
    return theta % (2*math.pi)

def make_checkpoints(x_min, x_max, count=None, explicit=None):
    """
    Generates a list of checkpoints for reporting progress and saving state.
    Checkpoints are spaced logarithmically for a more uniform distribution
    of reporting intervals over the prime number line.
    """
    if explicit:
        return sorted(set(int(float(s)) for s in explicit if x_min < float(s) <= x_max))
    if count is None:
        count = 60
    logs = np.linspace(math.log(x_min), math.log(x_max), count)
    return sorted(set(int(round(math.exp(v))) for v in logs if x_min < math.exp(v) <= x_max))

def circular_kde_fft(hist_counts, sigmas, bin_angles):
    """
    Performs a circular Kernel Density Estimation (KDE) using the Fast Fourier
    Transform (FFT). This is a computationally efficient method for smoothing
    the circular prime phase histogram and identifying peaks.
    
    Args:
        hist_counts: A NumPy array of the raw histogram counts.
        sigmas: A list of smoothing bandwidths (sigmas) to test.
        bin_angles: The angles corresponding to each histogram bin.
    
    Returns:
        A dictionary where each key is a sigma and the value is a tuple
        (smoothed_density_rho, standardized_z_score, peak_angle, max_z_score).
    """
    M = len(hist_counts)
    hist_fft = np.fft.fft(hist_counts.astype(float))
    h = np.fft.fftfreq(M, d=1.0/M)
    h2 = h*h
    out = {}
    for sigma in sigmas:
        kernel_fft = np.exp(-0.5 * (sigma**2) * h2)
        rho = np.fft.ifft(hist_fft * kernel_fft).real
        mu, sd = rho.mean(), rho.std(ddof=0)
        z = np.zeros_like(rho) if sd == 0.0 else (rho - mu)/sd
        idx = int(np.argmax(z))
        out[sigma] = (rho, z, bin_angles[idx], float(z[idx]))
    return out

def segmented_sieve(low, high):
    """
    A segmented sieve of Eratosthenes to find primes in the range [low, high].
    This approach is highly memory-efficient for large ranges because it does not
    require storing a global list of primes.
    
    Args:
        low: The start of the range (inclusive).
        high: The end of the range (inclusive).
    
    Returns:
        A NumPy array of primes in the specified range.
    """
    low = max(low, 2)
    size = high - low + 1
    sieve = np.ones(size, dtype=bool)
    limit = int(math.isqrt(high)) + 1
    base = np.ones(limit+1, dtype=bool); base[0:2] = False
    for i in range(2, int(math.isqrt(limit))+1):
        if base[i]:
            base[i*i:limit+1:i] = False
    base_primes = np.nonzero(base)[0]
    for p in base_primes:
        start = ((low+p-1)//p)*p
        if start < p*p: start = p*p
        sieve[start-low:size:p] = False
    if low == 2: sieve[0] = True
    return (low + np.nonzero(sieve)[0])

# --------------------------- State I/O: Checkpointing -------------------------------

def save_state(state_path, X_curr, reV, imV, reC, imC, Sg, Sc, hist_counts, meta):
    """Saves the current state of the run for checkpointing."""
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(state_path.with_suffix(".npy"), hist_counts)
    payload = {
        "X_curr": int(X_curr),
        "reV": float(reV), "imV": float(imV),
        "reC": float(reC), "imC": float(imC),
        "Sg": float(Sg), "Sc": float(Sc),
        "M": int(len(hist_counts)),
        "meta": meta,
    }
    state_path.write_text(json.dumps(payload))

def load_state(state_path):
    """Loads a previously saved state from disk."""
    payload = json.loads(Path(state_path).read_text())
    hist_counts = np.load(Path(state_path).with_suffix(".npy"))
    return payload, hist_counts

# ----------------------------- Main Runner Function --------------------------------

def run(X_max, segment, bins, sigmas,
        out_csv, state_file, checkpoints,
        resume=False, start_from=2,
        snapshot_dir=None, report_every=5.0,
        segment_timing=False, summary_txt=None,
        final_plots=None, plot_sigma=None):

    t0 = time.time()
    out_csv = Path(out_csv)
    if not out_csv.exists():
        # Write header for the output CSV file
        out_csv.write_text(
            "X,theta_star_deg,argV_deg,delta_star_vs_argV_deg,delta_star_vs_peak_deg,absV,ZV"
            + "".join([f",peak_deg_sigma_{s},maxZ_sigma_{s}" for s in sigmas])
            + ",gamma_standard,gamma_optimized\n"
        )

    if resume and Path(state_file).exists():
        # Resume a previous run from a saved state file
        payload, hist_counts = load_state(state_file)
        X_curr = payload["X_curr"]
        reV, imV = payload["reV"], payload["imV"]
        reC, imC = payload["reC"], payload["imC"]
        Sg, Sc = payload.get("Sg", 0.0), payload.get("Sc", 0.0)
        if payload["M"] != bins:
            raise ValueError(f"State bins ({payload['M']}) != requested bins ({bins}).")
        meta = payload["meta"]
        start_from = max(start_from, X_curr+1)
        print(f"[RESUME] Loaded state at X={X_curr}.")
    else:
        # Initialize a new run
        hist_counts = np.zeros(bins, dtype=np.int64)
        reV = imV = 0.0
        reC = imC = 0.0
        Sg = Sc = 0.0
        meta = {"twopi": 2.0*math.pi}
        X_curr = start_from-1

    dtheta = 2.0*math.pi/bins
    bin_angles = np.arange(bins)*dtheta
    cps = make_checkpoints(10**5, X_max, explicit=checkpoints)
    cp_idx = 0
    while cp_idx < len(cps) and cps[cp_idx] <= X_curr: cp_idx += 1

    processed_primes = 0
    last_report = time.time()

    X_list, gamma_std_vals, gamma_opt_vals = [], [], []

    def dist_to_180_deg(a, b):
        """Calculates the angular distance between two angles in degrees, centered around 180 degrees."""
        delta = (a - b)
        delta = ((delta + math.pi) % (2*math.pi)) - math.pi
        deg = abs(math.degrees(delta))
        return 180.0 - abs(180.0 - deg)

    for low in range(max(2, start_from), X_max+1, segment):
        seg_t0 = time.time() if segment_timing else None
        high = min(low+segment-1, X_max)
        primes = segmented_sieve(low, high)

        if primes.size:
            # Calculate the phase of each prime: phi(p) = ln(p) mod 2pi
            lnp = np.log(primes.astype(np.float64))
            phase = np.fmod(lnp, 2.0*math.pi)
            
            # Update the circular histogram
            idx = np.floor(phase/dtheta).astype(int)
            idx = np.where(idx>=bins, 0, idx)
            np.add.at(hist_counts, idx, 1)

            # Update the prime vector sum V(X) using Kahan summation
            invp = 1.0/primes.astype(np.float64)
            reV, reC = kahan_add(reV, reC, (np.cos(lnp)*invp).sum())
            imV, imC = kahan_add(imV, imC, (np.sin(lnp)*invp).sum())

            # Update the sum for the standard gamma estimator Sg(X)
            # The term is -log(1 - 1/p) - 1/p, which can be approximated
            # with log1p for better numerical stability.
            incr = (-np.log1p(-invp) - invp).sum()
            Sg, Sc = kahan_add(Sg, Sc, incr)

            processed_primes += primes.size
            X_curr = high

        if segment_timing:
            print(f"[SEG] {low:,}–{high:,} | +{primes.size:,} primes | {time.time()-seg_t0:.3f}s")

        while cp_idx < len(cps) and X_curr >= cps[cp_idx]:
            # Process a checkpoint to save results and report progress
            X_cp = cps[cp_idx]
            argV = math.atan2(imV, reV)
            theta_star = wrap_2pi(argV+math.pi)
            absV = math.hypot(reV, imV)
            
            # Z-score for the prime vector sum |V(X)|
            # This is benchmarked against the expected sqrt(log log x) growth.
            ZV = absV/math.sqrt(max(1e-12, math.log(math.log(max(3.0, X_cp)))))

            # Perform circular KDE via FFT
            kde = circular_kde_fft(hist_counts, sigmas, bin_angles)
            sigma0 = sigmas[0]
            _, z0, peak_angle, peakZ = kde[sigma0]

            # Calculate angular differences for consistency checks
            delta_star_vs_argV_deg  = dist_to_180_deg(theta_star, argV)
            delta_star_vs_peak_deg  = dist_to_180_deg(theta_star, peak_angle)

            # Calculate the standard and phase-optimized gamma estimators
            g_std = math.log(math.log(X_cp)) + Sg
            g_opt = math.log(math.log(X_cp)) + Sg + absV

            # Write the current checkpoint's data to the CSV
            row = [X_cp, math.degrees(theta_star), math.degrees(argV),
                   delta_star_vs_argV_deg, delta_star_vs_peak_deg,
                   absV, ZV]
            for s in sigmas:
                _, z, pk, pkZ = kde[s]
                row.extend([math.degrees(pk), pkZ])
            row.extend([g_std, g_opt])
            with out_csv.open("a") as f:
                f.write(",".join(str(v) for v in row)+"\n")

            if snapshot_dir:
                # Optional: Generate snapshot plots at checkpoints
                try:
                    import matplotlib.pyplot as plt
                    sd = Path(snapshot_dir); sd.mkdir(parents=True, exist_ok=True)
                    for s in sigmas:
                        rho, z, pk, pkZ = kde[s]
                        fig = plt.figure(figsize=(5,4))
                        plt.plot(bin_angles*180.0/math.pi, z)
                        plt.axvline(math.degrees(theta_star), ls="--", alpha=0.7, label=r"$\theta_X^\ast$")
                        plt.axvline(math.degrees(pk), ls=":", alpha=0.7, label=f"peak σ={s}")
                        plt.title(f"Z-score ρ at X={X_cp:,}, σ={s}")
                        plt.xlabel("θ (deg)"); plt.ylabel("Z(ρ)"); plt.legend()
                        fig.tight_layout()
                        fig.savefig(sd/f"zscore_X{X_cp}_sigma{s}.png", dpi=130)
                        plt.close(fig)
                except Exception as e:
                    print(f"[WARN] Snapshot plotting failed at X={X_cp}: {e}")

            save_state(state_file, X_curr, reV, imV, reC, imC, Sg, Sc, hist_counts, meta)
            X_list.append(X_cp); gamma_std_vals.append(g_std); gamma_opt_vals.append(g_opt)
            cp_idx += 1

        now = time.time()
        if now-last_report > report_every:
            # Print a progress report
            pct = (X_curr-start_from+1)/(X_max-start_from+1)*100.0
            theta_deg = (math.degrees(math.atan2(imV, reV)+math.pi))%360.0
            print(f"[{pct:5.1f}%] X≈{X_curr:,} | primes+={processed_primes:,} | θ*={theta_deg:6.2f}°")
            processed_primes = 0; last_report = now

    save_state(state_file, X_curr, reV, imV, reC, imC, Sg, Sc, hist_counts, meta)
    print(f"Done to X={X_curr:,} in {time.time()-t0:.1f}s. CSV -> {out_csv}")

    # -------- Summary & Final Plots (optional) --------
    if len(gamma_std_vals) >= 2 and (summary_txt or final_plots):
        true_gamma = 0.57721566490153286060651209
        std_err = np.array(gamma_std_vals) - true_gamma
        opt_err = np.array(gamma_opt_vals) - true_gamma
        var_std = float(np.var(std_err))
        var_opt = float(np.var(opt_err))
        red = (var_std / var_opt) if var_opt > 0 else float('inf')

        # Sigma for the final plots
        if plot_sigma is None:
            plot_sigma = sigmas[0]
        rho, z, pk_ang, pkZ = circular_kde_fft(hist_counts, [plot_sigma], bin_angles)[plot_sigma]
        max_z = float(pkZ)

        # ---- Summary .txt
        if summary_txt:
            txt = []
            txt.append("Running Hotspot analysis...\n\n")
            txt.append("Running Gamma estimator trajectory analysis...\n\n")
            txt.append(f"Results for X up to {X_curr:.0e}:\n\n")
            txt.append(f"Max Z-score (Hotspot Significance): {max_z:.4f}\n")
            txt.append(f"Standard Gamma Estimator Variance: {var_std:.9e}\n")
            txt.append(f"Optimized Gamma Estimator Variance: {var_opt:.9e}\n")
            if np.isfinite(red):
                txt.append(f"Variance Reduction Factor: {red:.2f}x\n")
            else:
                txt.append("Variance Reduction Factor: inf\n")
            Path(summary_txt).write_text("".join(txt))
            print(f"[OK] Summary written to {summary_txt}")

        # ---- Final multipanel plot
        if final_plots:
            try:
                import matplotlib.pyplot as plt
                outdir = Path(final_plots); outdir.mkdir(parents=True, exist_ok=True)

                fig, axes = plt.subplots(2, 3, figsize=(18, 12))

                # (1) Phase density ρ(θ) (polar)
                axp = plt.subplot(2, 3, 1, projection='polar')
                axp.plot(bin_angles, rho, linewidth=1.2)
                axp.set_title(f'Phase Density ρ(θ)\nσ={plot_sigma}, X={X_curr:.0e}')

                # (2) Z(θ) with theta* and Hotspot
                argV = math.atan2(imV, reV)
                theta_star = wrap_2pi(argV + math.pi)
                axes[0,1].plot(bin_angles, z, linewidth=1.2)
                axes[0,1].axvline(theta_star, linestyle='--', alpha=0.8, label=r"$\theta_X^\ast$")
                axes[0,1].axvline(pk_ang,    linestyle=':',  alpha=0.8, label='Hotspot')
                axes[0,1].set_xlabel('θ (rad)'); axes[0,1].set_ylabel('Z-score')
                axes[0,1].set_title('Standardized Phase Field'); axes[0,1].legend()
                axes[0,1].grid(True, alpha=0.3)

                # (3) Histogram density vs. Uniform
                dens = hist_counts / max(1, hist_counts.sum()) / (2*np.pi/bins)
                axes[0,2].plot(bin_angles, dens, alpha=0.85)
                axes[0,2].axhline(1/(2*np.pi), linestyle='--', alpha=0.7, label='Uniform')
                axes[0,2].axvline(pk_ang, linestyle=':', alpha=0.8, label='Hotspot')
                axes[0,2].set_xlabel('φ(p)'); axes[0,2].set_ylabel('Density')
                axes[0,2].set_title('Prime Phase Distribution'); axes[0,2].legend(); axes[0,2].grid(True, alpha=0.3)

                # (4) Gamma Trajectories
                axes[1,0].plot(X_list, gamma_std_vals, label='Standard', alpha=0.9)
                axes[1,0].plot(X_list, gamma_opt_vals, label='Phase-Optimized', alpha=0.9)
                axes[1,0].axhline(true_gamma, linestyle='--', alpha=0.7, label='True γ')
                axes[1,0].set_xlabel('X'); axes[1,0].set_ylabel('γ estimate')
                axes[1,0].set_title('Gamma Estimator Trajectories'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

                # (5) Error Trajectories
                axes[1,1].plot(X_list, std_err, label='Standard Error', alpha=0.9)
                axes[1,1].plot(X_list, opt_err, label='Optimized Error', alpha=0.9)
                axes[1,1].axhline(0, linestyle='--', alpha=0.7)
                axes[1,1].set_xlabel('X'); axes[1,1].set_ylabel('Error (estimate - true γ)')
                axes[1,1].set_title('Error Trajectories'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

                # (6) θ* vs. Hotspot (Degrees)
                axes[1,2].plot(bin_angles*180/np.pi, z, alpha=0.9)
                axes[1,2].axvline(math.degrees(theta_star), linestyle='--', alpha=0.8, label=r"$\theta_X^\ast$")
                axes[1,2].axvline(math.degrees(pk_ang),    linestyle=':',  alpha=0.8, label='Hotspot')
                axes[1,2].set_xlabel('θ (deg)'); axes[1,2].set_ylabel('Z-score')
                axes[1,2].set_title('θ* vs Hotspot (deg)'); axes[1,2].legend(); axes[1,2].grid(True, alpha=0.3)

                fig.tight_layout()
                outpath = outdir / f"prime_phase_summary_X{X_curr}.png"
                fig.savefig(outpath, dpi=130)
                plt.close(fig)
                print(f"[OK] Final plots written to {outpath}")
            except Exception as e:
                print(f"[WARN] Final plots failed: {e}")

# ------------------------------- CLI: Command Line Interface --------------------------------

def main():
    """
    Parses command-line arguments and initiates the prime phase analysis.
    This provides a flexible interface for running the script with different
    parameters (e.g., max X, segment size, resume from checkpoint).
    """
    ap = argparse.ArgumentParser(
        description="Streaming prime phase analysis up to X_max, with checkpointing.")
    ap.add_argument("--X_max", type=int, required=True,
                    help="Maximum number to sieve up to.")
    ap.add_argument("--segment", type=int, default=5_000_000,
                    help="Segment size for the segmented sieve.")
    ap.add_argument("--bins", type=int, default=65536,
                    help="Number of bins for the circular histogram.")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.05,0.1,0.2],
                    help="List of KDE bandwidths to analyze.")
    ap.add_argument("--out_csv", type=str, default="phase_drift.csv",
                    help="Output CSV file for checkpoint data.")
    ap.add_argument("--state_file", type=str, default="phase_state.json",
                    help="File for saving and loading the run state.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from a previous checkpoint if it exists.")
    ap.add_argument("--start_from", type=int, default=2,
                    help="Start sieving from this number (overrides resume if higher).")
    ap.add_argument("--checkpoints", type=str, nargs="*",
                    help="Explicit list of numbers for checkpointing.")
    ap.add_argument("--snapshots", type=str, default=None,
                    help="Directory to save snapshot plots at checkpoints.")
    ap.add_argument("--report_every", type=float, default=5.0,
                    help="Time interval (in seconds) between progress reports.")
    ap.add_argument("--segment_timing", action="store_true",
                    help="Print timing for each sieve segment.")
    ap.add_argument("--summary_txt", type=str, default=None,
                    help="Path to save the final summary text file.")
    ap.add_argument("--final_plots", type=str, default=None,
                    help="Directory to save the final multi-panel plot.")
    ap.add_argument("--plot_sigma", type=float, default=None,
                    help="Specific sigma to use for final plots.")
    args = ap.parse_args()

    run(args.X_max, args.segment, args.bins, args.sigmas,
        args.out_csv, args.state_file, args.checkpoints,
        resume=args.resume, start_from=args.start_from,
        snapshot_dir=args.snapshots, report_every=args.report_every,
        segment_timing=args.segment_timing,
        summary_txt=args.summary_txt, final_plots=args.final_plots,
        plot_sigma=args.plot_sigma)

if __name__=="__main__":
    main()