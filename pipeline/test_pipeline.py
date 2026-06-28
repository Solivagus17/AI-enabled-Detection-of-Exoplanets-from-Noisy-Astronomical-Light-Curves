"""
ExoHunter Test Suite
Author: Sarthakk Anjariya

This test suite verifies the functionality of all exoplanet pipeline components:
1. Ingestion (fetch a single test star: Pi Mensae c / TOI-144)
2. Preprocessing (outliers, flattening, gap-filling, normalization, padding to 4000 pts)
3. Candidate Detection (BLS search, peaks, folding, binning to 200 pts)
4. Parameter Estimation (depth, duration, SNR, bootstrap 1-sigma uncertainty)
5. Model architectures (CNN-LSTM and Autoencoder compilation)

Usage:
  python -m unittest pipeline/test_pipeline.py
"""

import os
import sys
import unittest
import numpy as np
import tensorflow as tf

# Add local path to import modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from ingest import fetch
from preprocess import preprocess_single
from detect import detect_candidates
from classify import build_cnn_lstm, build_autoencoder
from estimate import estimate_parameters

class TestExoplanetPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_tic = "TIC 150428135"  # Pi Mensae c (known exoplanet)
        cls.raw_dir = os.path.join(BASE_DIR, "data", "raw")
        cls.processed_dir = os.path.join(BASE_DIR, "data", "processed")
        
        # Ensure directories exist
        os.makedirs(cls.raw_dir, exist_ok=True)
        os.makedirs(cls.processed_dir, exist_ok=True)
        
        # Download the test TIC light curve for verification
        print(f"\n[Test Setup] Downloading light curve for test target {cls.test_tic}...")
        cls.fits_path = fetch(cls.test_tic, cache_dir=cls.raw_dir)

    def test_1_ingest(self):
        """Test if the FITS file ingestion works and caches the file correctly."""
        self.assertIsNotNone(self.fits_path, "FITS file download returned None")
        self.assertTrue(os.path.exists(self.fits_path), f"FITS file does not exist at {self.fits_path}")
        self.assertTrue(os.path.getsize(self.fits_path) > 0, "FITS file is empty (0 bytes)")
        print(f"[Test Pass] Ingest: Cached FITS file exists at {self.fits_path}")

    def test_2_preprocess(self):
        """Test if preprocessing outputs a normalized flux array of shape (4000,)."""
        self.assertTrue(os.path.exists(self.fits_path), "FITS file not found for preprocessing test")
        flux, time = preprocess_single(self.fits_path, target_length=4000)
        
        self.assertIsNotNone(flux, "Preprocessing returned None for flux")
        self.assertIsNotNone(time, "Preprocessing returned None for time")
        self.assertEqual(flux.shape, (4000,), f"Flux shape should be (4000,), but got {flux.shape}")
        self.assertEqual(time.shape, (4000,), f"Time shape should be (4000,), but got {time.shape}")
        
        # Confirm normalization (median flux should be close to 1.0)
        # Note: raw values might be padded, check median of non-padded or entire array
        self.assertAlmostEqual(np.median(flux), 1.0, places=2, msg="Preprocessed flux is not normalized to 1.0")
        print(f"[Test Pass] Preprocess: Uniform light curve size and normalization verified.")

    def test_3_detect(self):
        """Test if BLS runs, yields candidate periods, and folds them to (200, 1)."""
        flux, time = preprocess_single(self.fits_path, target_length=4000)
        candidates = detect_candidates(time, flux)
        
        self.assertTrue(len(candidates) >= 1, "No candidate periods found")
        self.assertEqual(len(candidates), 3, f"Expected 3 candidates, got {len(candidates)}")
        
        # Validate candidate structure
        best_cand = candidates[0]
        self.assertIn("period", best_cand)
        self.assertIn("snr", best_cand)
        self.assertIn("folded_flux", best_cand)
        self.assertEqual(best_cand["folded_flux"].shape, (200, 1), "Folded flux shape must be (200, 1)")
        self.assertEqual(best_cand["folded_phase"].shape, (200, 1), "Folded phase shape must be (200, 1)")
        print(f"[Test Pass] Detect: BLS peak identification and (200, 1) phase folding verified.")

    def test_4_models(self):
        """Test if CNN-LSTM classifier and Autoencoder compile with correct shapes."""
        clf = build_cnn_lstm(input_shape=(200, 1))
        ae = build_autoencoder(input_shape=(200, 1))
        
        # Check shapes
        self.assertEqual(clf.input_shape, (None, 200, 1), "Classifier input shape mismatch")
        self.assertEqual(clf.output_shape, (None, 4), "Classifier output shape must be (None, 4)")
        
        self.assertEqual(ae.input_shape, (None, 200, 1), "Autoencoder input shape mismatch")
        self.assertEqual(ae.output_shape, (None, 200, 1), "Autoencoder output shape must be (None, 200, 1)")
        print(f"[Test Pass] Classify: Deep Learning architectures compiled with correct dimensions.")

    def test_5_estimate(self):
        """Test if parameter estimation returns correct keys, types, and bootstrap bounds."""
        flux, time = preprocess_single(self.fits_path, target_length=4000)
        candidates = detect_candidates(time, flux)
        best_cand = candidates[0]
        
        # Execute estimation
        # Skip neural network inputs to test parameter calculation and bootstrapping separately
        res = estimate_parameters(time, flux, best_cand, clf_model=None, ae_model=None, class_label="transit")
        
        # Verify output keys and ranges
        self.assertEqual(res["class"], "transit")
        self.assertTrue(res["period_days"] > 0)
        self.assertTrue(res["depth_ppm"] > 0)
        self.assertTrue(res["duration_hrs"] > 0)
        
        # Verify uncertainty estimates are non-zero
        self.assertTrue(res["period_err"] > 0, "Period error cannot be 0")
        self.assertTrue(res["depth_err"] > 0, "Depth error cannot be 0")
        self.assertTrue(res["duration_err"] > 0, "Duration error cannot be 0")
        print(f"[Test Pass] Estimate: Parameter estimates and 100x bootstrap uncertainties calculated.")

if __name__ == "__main__":
    unittest.main()
