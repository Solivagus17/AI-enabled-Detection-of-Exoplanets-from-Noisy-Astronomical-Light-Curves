"""
ExoHunter Detection Module
Author: Antigravity

This module implements Box Least Squares (BLS) periodogram search to detect
periodic, box-shaped transit signals in preprocessed light curves.

Astrophysical reasoning:
- Planets transit their host stars periodically. BLS searches a grid of periods
  and transit durations, fitting a step-function (box) dip to the light curve.
- The BLS algorithm is optimal for detecting shallow, periodic transits where
  the out-of-transit flux is flat and the in-transit flux is lower by a constant depth.
- The standard astrophysics detection threshold is SNR > 7.1 (originally defined
  in the Kepler mission based on false alarm rates in white noise).
"""

import os
import logging
import numpy as np
from astropy.timeseries import BoxLeastSquares
import lightkurve as lk
from scipy.signal import find_peaks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def bin_folded_lc(time_offsets, fluxes, period, n_bins=200):
    """
    Manually bins a phase-folded light curve into a fixed number of bins.
    Assumes the time_offsets range from -period/2 to +period/2.
    
    Parameters:
    - time_offsets: np.ndarray, time offsets from the transit midpoint
    - fluxes: np.ndarray, normalized flux values
    - period: float, period in days
    - n_bins: int, number of phase bins
    
    Returns:
    - bin_centers: np.ndarray, center of each phase bin (from -0.5 to 0.5)
    - binned_flux: np.ndarray, median flux in each bin, shape (n_bins,)
    - binned_err: np.ndarray, standard error of flux in each bin, shape (n_bins,)
    """
    phases = time_offsets / period  # normalized phase from -0.5 to 0.5
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    binned_flux = np.ones(n_bins, dtype=np.float32)
    binned_err = np.zeros(n_bins, dtype=np.float32)
    
    for i in range(n_bins):
        mask = (phases >= bin_edges[i]) & (phases < bin_edges[i+1])
        if np.any(mask):
            binned_flux[i] = np.median(fluxes[mask])
            # Calculate standard error: std / sqrt(N)
            n_in_bin = np.sum(mask)
            if n_in_bin > 1:
                binned_err[i] = np.std(fluxes[mask]) / np.sqrt(n_in_bin)
            else:
                binned_err[i] = 0.0
        else:
            binned_flux[i] = 1.0  # Default flat baseline
            binned_err[i] = 0.0
            
    return bin_centers, binned_flux, binned_err

def detect_candidates(time, flux, min_period=0.5, max_period=15.0):
    """
    Search for periodic transit signals using BLS and return the top-3 period candidates,
    along with their folded phase curves (200 points) and SNR.
    
    Parameters:
    - time: np.ndarray, time values in days
    - flux: np.ndarray, normalized, flattened flux values
    - min_period: float, minimum period to search in days
    - max_period: float, maximum period to search in days
    
    Returns:
    - list of dicts: [
        {
          "period": float,
          "transit_time": float,
          "duration": float,
          "depth": float,
          "snr": float,
          "folded_phase": np.ndarray,  # shape (200, 1)
          "folded_flux": np.ndarray,   # shape (200, 1)
          "flagged": bool
        }, ...
      ]
    """
    # Initialize astropy BLS
    bls = BoxLeastSquares(time, flux)
    
    # Define duration grid: from 1 hour (0.04 days) to 5 hours (0.21 days)
    durations = np.linspace(0.04, 0.21, 10)
    
    # Auto-grid for periods
    period_grid = bls.autoperiod(durations, minimum_period=min_period, maximum_period=max_period)
    
    # Compute power spectrum
    results = bls.power(period_grid, durations)
    
    # Find peaks in the BLS power spectrum
    power = results.power
    periods = results.period
    
    # Identify local peaks in the power spectrum using find_peaks
    # We require peaks to be separated by at least 20 grid points
    peaks, _ = find_peaks(power, distance=20)
    
    # Sort peaks by power in descending order
    if len(peaks) > 0:
        peaks = peaks[np.argsort(power[peaks])][::-1]
        top_periods = periods[peaks[:3]]
    else:
        # Fallback: take the absolute max and its sub-harmonics
        top_periods = []
        
    # If we don't have 3 peaks, fill using sorted powers
    if len(top_periods) < 3:
        sorted_idx = np.argsort(power)[::-1]
        for idx in sorted_idx:
            p = periods[idx]
            # Avoid duplicate periods close to already selected ones
            if not any(np.isclose(p, tp, rtol=0.02) for tp in top_periods):
                top_periods = np.append(top_periods, p)
            if len(top_periods) == 3:
                break
                
    candidates = []
    
    # For each candidate period, compute the exact transit parameters and fold the light curve
    for p in top_periods:
        # Re-compute BLS stats for this specific period
        p_idx = np.argmin(np.abs(results.period - p))
        best_duration = results.duration[p_idx]
        best_transit_time = results.transit_time[p_idx]
        best_depth = results.depth[p_idx]
        
        # Fold the light curve using the best period and transit midpoint time (epoch)
        # We manually compute the time offsets relative to the nearest transit midpoint
        # so that transit is centered at 0
        offsets = (time - best_transit_time + 0.5 * p) % p - 0.5 * p
        
        # Bin the folded light curve to 200 bins
        bin_centers, binned_flux, binned_err = bin_folded_lc(offsets, flux, p, n_bins=200)
        
        # Compute RMS noise in out-of-transit bins (outer 80% of phase space)
        # Phase goes from -0.5 to 0.5, so phase indices < 40 and > 160 are out-of-transit
        out_transit_mask = np.ones(200, dtype=bool)
        out_transit_mask[40:160] = False
        
        rms_noise = np.std(binned_flux[out_transit_mask])
        if rms_noise == 0 or np.isnan(rms_noise):
            # Fallback to robust standard deviation (MAD) of the raw flux
            rms_noise = 1.4826 * np.median(np.abs(flux - np.median(flux)))
            
        # Recalculate depth as 1.0 - min(binned_flux)
        depth = 1.0 - np.min(binned_flux)
        snr = depth / rms_noise if rms_noise > 0 else 0.0
        
        flagged = snr > 7.1
        
        candidates.append({
            "period": float(p),
            "transit_time": float(best_transit_time),
            "duration": float(best_duration),
            "depth": float(depth),
            "snr": float(snr),
            "folded_phase": bin_centers.reshape(-1, 1),
            "folded_flux": binned_flux.reshape(-1, 1),
            "flagged": flagged
        })
        
    # Sort candidates by SNR in descending order
    candidates = sorted(candidates, key=lambda x: x["snr"], reverse=True)
    return candidates

if __name__ == "__main__":
    # Test on a preprocessed file if it exists
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "data", "processed")
    X_path = os.path.join(processed_dir, "X.npy")
    times_path = os.path.join(processed_dir, "times.npy")
    tics_path = os.path.join(processed_dir, "tics.npy")
    
    if os.path.exists(X_path) and os.path.exists(times_path):
        X = np.load(X_path)
        times = np.load(times_path)
        tics = np.load(tics_path, allow_pickle=True)
        
        logging.info("Testing detect_candidates on first light curve...")
        # Check first object
        flux = X[0, :, 0]
        time = times[0]
        tic = tics[0]
        
        candidates = detect_candidates(time, flux)
        for i, c in enumerate(candidates):
            logging.info(f"Candidate {i+1} for {tic}: Period={c['period']:.4f} days, SNR={c['snr']:.2f}, Flagged={c['flagged']}")
    else:
        logging.info("Preprocessed data not found. Run ingest.py and preprocess.py first.")
