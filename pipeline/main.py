"""
ExoHunter Orchestration Module
Author: Sarthakk Anjariya

This is the main CLI entry point for the ExoHunter exoplanet detection pipeline.
It orchestrates the stages of:
- Data Ingestion (--ingest)
- Preprocessing (--preprocess)
- Neural Network Training (--train)
- Single-Star Inference (--tic "TIC 261136679")
- Batch Prediction on all TIC IDs (--predict-all)

Usage examples:
  python main.py --run-all
  python main.py --tic "TIC 261136679"
  python main.py --predict-all
"""

import os
import sys
import argparse
import json
import logging
import numpy as np
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Add local path to import modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from ingest import batch_fetch, TIC_IDS, fetch
from preprocess import preprocess_all, preprocess_single
from detect import detect_candidates
from classify import train_and_evaluate
from estimate import estimate_parameters

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FAILED_LOG = os.path.join(BASE_DIR, "failed.txt")
PREDICTIONS_JSON = os.path.join(BASE_DIR, "data", "predictions.json")

def load_models():
    """
    Load trained classifier and autoencoder models.
    """
    clf_path = os.path.join(MODELS_DIR, "clf_model.keras")
    ae_path = os.path.join(MODELS_DIR, "ae_model.keras")
    
    clf = None
    ae = None
    
    if os.path.exists(clf_path):
        try:
            clf = tf.keras.models.load_model(clf_path)
            logging.info(f"Loaded classifier from {clf_path}")
        except Exception as e:
            logging.error(f"Error loading classifier: {e}")
            
    if os.path.exists(ae_path):
        try:
            ae = tf.keras.models.load_model(ae_path)
            logging.info(f"Loaded autoencoder from {ae_path}")
        except Exception as e:
            logging.error(f"Error loading autoencoder: {e}")
            
    return clf, ae

def run_single_tic(tic_id, clf_model=None, ae_model=None):
    """
    Run the complete ingestion, preprocessing, detection, classification,
    and parameter estimation pipeline for a single TIC ID.
    
    Returns:
    - dict, final predictions in target format
    """
    logging.info(f"=== Running pipeline for {tic_id} ===")
    
    # 1. Fetch data
    fits_path = fetch(tic_id, cache_dir=RAW_DIR)
    if fits_path is None or not os.path.exists(fits_path):
        logging.error(f"Failed to fetch data for {tic_id}")
        return None
        
    # 2. Preprocess
    flux, time = preprocess_single(fits_path)
    if flux is None or time is None:
        logging.error(f"Preprocessing failed for {tic_id}")
        return None
        
    # 3. Detect candidate transits
    candidates = detect_candidates(time, flux)
    if len(candidates) == 0:
        logging.error(f"No periodic candidates detected for {tic_id}")
        return None
        
    best_cand = candidates[0]
    
    # 4. Predict and estimate parameters
    # Try to find the true category from TIC_IDS for label reference
    true_label = "other"
    for cat, tics in TIC_IDS.items():
        if tic_id in tics:
            true_label = cat
            break
            
    result = estimate_parameters(
        time, flux, best_cand,
        clf_model=clf_model,
        ae_model=ae_model,
        class_label=true_label
    )
    result["tic_id"] = tic_id
    
    return result

def run_predict_all(clf_model, ae_model):
    """
    Run pipeline on all 100 TIC IDs, generate predictions, and save to JSON.
    """
    all_targets = []
    for cat, list_ids in TIC_IDS.items():
        for tic in list_ids:
            all_targets.append(tic)
            
    results = {}
    success_count = 0
    
    logging.info(f"Starting batch prediction for {len(all_targets)} TIC IDs...")
    
    for tic in all_targets:
        try:
            res = run_single_tic(tic, clf_model=clf_model, ae_model=ae_model)
            if res is not None:
                results[tic] = res
                success_count += 1
                # Log JSON output per star
                logging.info(f"Prediction for {tic}: {json.dumps(res, indent=2)}")
        except Exception as e:
            logging.error(f"Error running pipeline for {tic}: {e}")
            
    # Save to file
    os.makedirs(os.path.dirname(PREDICTIONS_JSON), exist_ok=True)
    with open(PREDICTIONS_JSON, "w") as f:
        json.dump(results, f, indent=4)
        
    logging.info(f"Batch prediction complete. Saved predictions for {success_count} stars to {PREDICTIONS_JSON}")
    return results

def main():
    parser = argparse.ArgumentParser(description="ExoHunter: TESS Transit Detector Pipeline")
    parser.add_argument("--ingest", action="store_true", help="Download and cache FITS files for the 100 TIC IDs")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess all FITS files and save numpy arrays")
    parser.add_argument("--train", action="store_true", help="Train CNN-LSTM and Autoencoder models")
    parser.add_argument("--run-all", action="store_true", help="Run ingest, preprocess, and train stages sequentially")
    parser.add_argument("--tic", type=str, help="Process and run inference on a single TIC ID, e.g., 'TIC 261136679'")
    parser.add_argument("--predict-all", action="store_true", help="Run predictions on all 100 stars and save to JSON")
    
    args = parser.parse_args()
    
    # If no arguments provided, print help
    if len(sys.argv) == 1:
        parser.print_help()
        return

    # Check directories
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.run_all or args.ingest:
        logging.info("--- Stage 1: Data Ingestion ---")
        batch_fetch(cache_dir=RAW_DIR, failed_log=FAILED_LOG)
        
    if args.run_all or args.preprocess:
        logging.info("--- Stage 2: Data Preprocessing ---")
        preprocess_all(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)
        
    if args.run_all or args.train:
        logging.info("--- Stage 3: Neural Network Training ---")
        train_and_evaluate(processed_dir=PROCESSED_DIR, models_dir=MODELS_DIR)

    # Inferences
    if args.tic or args.predict_all:
        clf_model, ae_model = load_models()
        if clf_model is None or ae_model is None:
            logging.warning("Note: Classifications and anomaly scores will fall back to default values because trained neural network models were not found. Run with --train to train them.")
            
        if args.tic:
            res = run_single_tic(args.tic, clf_model=clf_model, ae_model=ae_model)
            if res is not None:
                print("\n" + "="*50)
                print("PIPELINE RESULT PER STAR:")
                print("="*50)
                print(json.dumps(res, indent=4))
                print("="*50 + "\n")
                
        if args.predict_all:
            run_predict_all(clf_model, ae_model)

if __name__ == "__main__":
    main()
