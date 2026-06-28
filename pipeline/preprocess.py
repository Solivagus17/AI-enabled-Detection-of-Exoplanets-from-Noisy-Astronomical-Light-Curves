"""
ExoHunter Preprocessing Module
Author: Sarthakk Anjariya

This module processes raw TESS FITS light curves into a clean, uniform, and standardized
format ready for periodic signal detection (BLS) and deep learning classification.

Astrophysical reasoning:
- Outlier Removal: Cosmic rays and spacecraft pointing jitter can create spurious spikes.
  Sigma-clipping at 3-sigma removes these non-astrophysical artifacts.
- Flattening: Stellar rotation, starspots, and instrumental trends introduce slow variation.
  A Savitzky-Golay filter (implemented in Lightkurve's .flatten with window_length=401)
  removes these long-term trends, keeping the short-duration transit features intact.
- Normalization: Divides by the median flux so that the out-of-transit baseline is exactly 1.0.
  This allows depths to be compared directly across different stars of different magnitudes.
- Fixed-length array (4000 pts): Standardizes input size for downstream deep learning networks.
  We pad with 1.0 because a flattened, normalized out-of-transit light curve has a flux baseline of 1.0.
"""

import os
import logging
import numpy as np
import lightkurve as lk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def preprocess_single(fits_path, target_length=4000):
    """
    Load a raw TESS FITS file, remove outliers, flatten, fill gaps, normalize,
    and pad/truncate to a fixed length of 4000 points.
    
    Parameters:
    - fits_path: str, path to the FITS file
    - target_length: int, output number of data points
    
    Returns:
    - flux: np.ndarray, shape (target_length,), preprocessed flux values
    - time: np.ndarray, shape (target_length,), corresponding time values
    """
    try:
        # Load the light curve
        lc = lk.read(fits_path)
        
        # 1. Remove outliers (3-sigma clipping)
        lc = lc.remove_outliers(sigma=3)
        
        # 2. Flatten stellar variability/instrumental trends
        # Note: window_length should be an odd integer. 401 is standard.
        # If the light curve is very short, we adjust window_length to be smaller
        window_len = 401
        if len(lc) <= window_len:
            window_len = (len(lc) // 2) * 2 - 1  # closest odd number smaller than len(lc)
            if window_len < 3:
                window_len = 3
        
        lc = lc.flatten(window_length=window_len)
        
        # 3. Fill gaps in the observation timeseries
        lc = lc.fill_gaps()
        
        # 4. Normalize the light curve (sets out-of-transit baseline to 1.0)
        lc = lc.normalize()
        
        flux = lc.flux.value
        time = lc.time.value
        
        # 5. Pad or truncate to target_length
        n_pts = len(flux)
        if n_pts >= target_length:
            # Slicing from the start (or we could take the middle)
            flux = flux[:target_length]
            time = time[:target_length]
        else:
            # Pad with 1.0 for normalized flux and extrapolate time linearly
            pad_size = target_length - n_pts
            
            # Pad flux with 1.0
            flux = np.pad(flux, (0, pad_size), mode='constant', constant_values=1.0)
            
            # Extrapolate time values linearly
            if n_pts > 1:
                dt = np.median(np.diff(time))
            else:
                dt = 0.00138  # TESS SPOC 2-minute cadence in days (~120 seconds)
                
            last_time = time[-1] if n_pts > 0 else 0.0
            padded_time = np.array([last_time + (i + 1) * dt for i in range(pad_size)])
            time = np.concatenate([time, padded_time])
            
        return flux, time
        
    except Exception as e:
        logging.error(f"Error preprocessing FITS file {fits_path}: {str(e)}")
        return None, None

def preprocess_all(raw_dir="data/raw", processed_dir="data/processed"):
    """
    Process all raw FITS files in raw_dir and save the output arrays in processed_dir.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    # Import TIC_IDS to match filenames and map labels
    from ingest import TIC_IDS
    
    label_map = {
        "transit": 0,
        "eclipse": 1,
        "blend": 2,
        "other": 3
    }
    
    X = []
    y = []
    tics = []
    classes = []
    times = []
    
    # Find all cached FITS files
    files = [f for f in os.listdir(raw_dir) if f.endswith(".fits")]
    logging.info(f"Found {len(files)} cached FITS files for preprocessing.")
    
    # Map TIC IDs to their category
    tic_to_class = {}
    for class_name, tic_list in TIC_IDS.items():
        for tic in tic_list:
            clean_tic = tic.replace(' ', '_')
            tic_to_class[clean_tic] = class_name
            
    success_count = 0
    
    for filename in files:
        tic_key = filename.replace('.fits', '')
        fits_path = os.path.join(raw_dir, filename)
        
        # Determine the label
        class_name = tic_to_class.get(tic_key, "other")
        label = label_map[class_name]
        
        # Preprocess
        flux, time = preprocess_single(fits_path)
        
        if flux is not None:
            X.append(flux)
            y.append(label)
            tics.append(tic_key.replace('_', ' '))
            classes.append(class_name)
            times.append(time)
            success_count += 1
            
    if success_count > 0:
        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)  # shape (N, 4000)
        y = np.array(y, dtype=np.int32)    # shape (N,)
        tics = np.array(tics, dtype=object)
        classes = np.array(classes, dtype=object)
        times = np.array(times, dtype=np.float32)
        
        # Add channel dimension to X: shape (N, 4000, 1)
        X_reshaped = np.expand_dims(X, axis=-1)
        
        # Save preprocessed files
        np.save(os.path.join(processed_dir, "X.npy"), X_reshaped)
        np.save(os.path.join(processed_dir, "y.npy"), y)
        np.save(os.path.join(processed_dir, "tics.npy"), tics)
        np.save(os.path.join(processed_dir, "classes.npy"), classes)
        np.save(os.path.join(processed_dir, "times.npy"), times)
        
        logging.info(f"Preprocessing complete. Successfully processed {success_count} light curves.")
        logging.info(f"Saved dataset shapes - X: {X_reshaped.shape}, y: {y.shape}")
    else:
        logging.error("No light curves were successfully preprocessed.")
        
    return success_count

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    preprocess_all(raw_dir=raw_dir, processed_dir=processed_dir)
