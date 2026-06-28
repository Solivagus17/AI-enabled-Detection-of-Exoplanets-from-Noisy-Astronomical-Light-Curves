"""
ExoHunter Streamlit Dashboard Application
Author: Antigravity

This is the main interactive interface for the exoplanet detection pipeline. It allows
users to load TESS light curves (either from the 100 hardcoded TIC IDs or custom FITS uploads),
run the processing pipeline, visualize the raw/folded light curves, and inspect detection
parameters, classification predictions, and anomaly scores.
"""

import os
import sys
import streamlit as st
import numpy as np
import tensorflow as tf
import lightkurve as lk

# Add local path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingest import TIC_IDS, fetch
from preprocess import preprocess_single
from detect import detect_candidates
from classify import train_and_evaluate
from estimate import estimate_parameters
from visualize import plot_raw_lc, plot_folded_lc

# Set page configuration with dark theme styling
st.set_page_config(
    page_title="ExoHunter — TESS Transit Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS
st.markdown("""
    <style>
        .reportview-container {
            background: #111726;
            color: #FFFFFF;
        }
        .sidebar .sidebar-content {
            background: #1A2238;
        }
        div[data-testid="stMetricValue"] {
            font-size: 24px;
            color: #FF7F0E;
            font-weight: bold;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 14px;
            color: #A0AEC0;
        }
        .main-header {
            font-size: 38px;
            font-weight: bold;
            background: -webkit-linear-gradient(45deg, #FF7F0E, #EF553B, #636EFA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
        }
        .card {
            background-color: #1A2238;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #2D3748;
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# App directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load models helper
@st.cache_resource
def load_deep_models():
    clf_path = os.path.join(MODELS_DIR, "clf_model.keras")
    ae_path = os.path.join(MODELS_DIR, "ae_model.keras")
    
    clf = None
    ae = None
    
    if os.path.exists(clf_path):
        try:
            clf = tf.keras.models.load_model(clf_path)
        except Exception as e:
            st.sidebar.error(f"Error loading classifier: {e}")
            
    if os.path.exists(ae_path):
        try:
            ae = tf.keras.models.load_model(ae_path)
        except Exception as e:
            st.sidebar.error(f"Error loading autoencoder: {e}")
            
    return clf, ae

clf_model, ae_model = load_deep_models()

# Sidebar Layout
st.sidebar.image("https://img.icons8.com/color/96/000000/telescope.png", width=80)
st.sidebar.title("ExoHunter Controls")

# Mode Selection
source_mode = st.sidebar.radio("Data Source", ["Preloaded 100 TIC IDs", "Upload Custom FITS File"])

selected_tic = None
uploaded_file_path = None
true_class = "other"

if source_mode == "Preloaded 100 TIC IDs":
    # Class Filter
    class_filter = st.sidebar.selectbox("Filter TICs by Category", ["All", "Transit", "Eclipse", "Blend", "Other"])
    
    # Get filtered TIC list
    filtered_tics = []
    if class_filter == "All":
        for cat, list_ids in TIC_IDS.items():
            filtered_tics.extend([(tic, cat) for tic in list_ids])
    else:
        cat_key = class_filter.lower()
        filtered_tics = [(tic, cat_key) for tic in TIC_IDS[cat_key]]
        
    tic_options = [f"{tic} ({cat})" for tic, cat in filtered_tics]
    selected_option = st.sidebar.selectbox("Select Target TIC ID", tic_options)
    
    if selected_option:
        # Extract tic and category
        selected_tic = selected_option.split(" (")[0]
        true_class = selected_option.split(" (")[1][:-1]
else:
    uploaded_file = st.sidebar.file_uploader("Upload TESS Light Curve (.fits)", type=["fits"])
    if uploaded_file is not None:
        # Save uploaded file
        os.makedirs(RAW_DIR, exist_ok=True)
        uploaded_file_path = os.path.join(RAW_DIR, "custom_target.fits")
        with open(uploaded_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        selected_tic = "Uploaded Target"
        true_class = "unknown"

# Controls
snr_threshold = st.sidebar.slider("SNR Detection Threshold", 3.0, 15.0, 7.1, 0.1)

# Model status and training button
st.sidebar.markdown("---")
st.sidebar.subheader("Model Configuration")
if clf_model is None or ae_model is None:
    st.sidebar.warning("Deep learning models not found.")
    if st.sidebar.button("Train Models Now"):
        with st.spinner("Training CNN-LSTM & Autoencoder... (takes ~1-2 mins)"):
            try:
                # First run ingest and preprocess if they haven't run
                # (Assuming files are present or we fetch them)
                train_and_evaluate(processed_dir=PROCESSED_DIR, models_dir=MODELS_DIR)
                st.sidebar.success("Training complete! Please reload the page.")
                st.cache_resource.clear()
            except Exception as e:
                st.sidebar.error(f"Training failed: {e}")
else:
    st.sidebar.success("Neural networks loaded successfully.")
    if st.sidebar.button("Retrain Models"):
        with st.spinner("Retraining..."):
            try:
                train_and_evaluate(processed_dir=PROCESSED_DIR, models_dir=MODELS_DIR)
                st.sidebar.success("Retraining complete! Please reload the page.")
                st.cache_resource.clear()
            except Exception as e:
                st.sidebar.error(f"Training failed: {e}")

# Main Layout
st.markdown('<div class="main-header">ExoHunter — TESS Planet Transit Detector</div>', unsafe_allow_html=True)
st.markdown("An artificial intelligence pipeline for detecting and classifying exoplanetary transits from Kepler/TESS space mission light curves.")

if selected_tic is None and uploaded_file_path is None:
    st.info("Please select a preloaded TIC ID or upload a FITS file from the sidebar to begin processing.")
else:
    with st.spinner(f"Running pipeline on {selected_tic}..."):
        # 1. Fetch file if preloaded
        fits_path = None
        if source_mode == "Preloaded 100 TIC IDs":
            fits_path = fetch(selected_tic, cache_dir=RAW_DIR)
        else:
            fits_path = uploaded_file_path
            
        if fits_path is None or not os.path.exists(fits_path):
            st.error(f"Could not load data for {selected_tic}. The file is missing or download failed.")
        else:
            # 2. Preprocess light curve
            flux_prep, time_prep = preprocess_single(fits_path)
            
            if flux_prep is None or time_prep is None:
                st.error("Preprocessing failed. FITS file might be corrupted or in an unexpected format.")
            else:
                # 3. Detect candidate transits using BLS
                candidates = detect_candidates(time_prep, flux_prep)
                
                # Filter candidates by SNR slider
                valid_candidates = [c for c in candidates if c["snr"] >= snr_threshold]
                
                if len(valid_candidates) == 0:
                    st.warning(f"No periodic signals found exceeding the SNR threshold of {snr_threshold:.1f}. Showing best candidate below.")
                    best_cand = candidates[0]
                else:
                    best_cand = valid_candidates[0]
                    
                # 4. Estimate parameters & classify
                params = estimate_parameters(
                    time_prep, flux_prep, best_cand, 
                    clf_model=clf_model, ae_model=ae_model, 
                    class_label=true_class
                )
                params["tic_id"] = selected_tic
                
                # Check SNR threshold for flagging
                is_flagged = params["snr"] >= snr_threshold
                
                # Main Dashboard Grid
                # Row 1: Metrics
                st.markdown("### Detection Metrics")
                mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
                
                with mcol1:
                    st.metric(
                        label="Classification", 
                        value=params["class"].upper(), 
                        delta=f"Conf: {params['confidence']*100:.1f}%" if clf_model is not None else None
                    )
                with mcol2:
                    st.metric(
                        label="Orbital Period", 
                        value=f"{params['period_days']:.4f} d", 
                        delta=f"±{params['period_err']:.4f} d",
                        delta_color="off"
                    )
                with mcol3:
                    st.metric(
                        label="Transit Depth", 
                        value=f"{params['depth_ppm']:,} ppm", 
                        delta=f"±{params['depth_err']} ppm",
                        delta_color="off"
                    )
                with mcol4:
                    st.metric(
                        label="Transit Duration", 
                        value=f"{params['duration_hrs']:.2f} hrs", 
                        delta=f"±{params['duration_err']:.2f} hrs",
                        delta_color="off"
                    )
                with mcol5:
                    st.metric(
                        label="Signal-to-Noise (SNR)", 
                        value=f"{params['snr']:.2f}",
                        delta="Flagged (Candidate)" if is_flagged else "Below Threshold",
                        delta_color="normal" if is_flagged else "inverse"
                    )
                    
                # Row 2: Plots side by side
                st.markdown("---")
                st.markdown("### Light Curve Visualization")
                pcol1, pcol2 = st.columns(2)
                
                # Duration in days for highlight box
                duration_days = (params["duration_hrs"] / 24.0)
                
                with pcol1:
                    fig_raw = plot_raw_lc(
                        time_prep, flux_prep, selected_tic, 
                        period=params["period_days"], 
                        epoch=best_cand["transit_time"], 
                        duration_days=duration_days
                    )
                    st.plotly_chart(fig_raw, use_container_width=True)
                    
                with pcol2:
                    # Plot folded curve
                    phases = best_cand["folded_phase"][:, 0]
                    fluxes = best_cand["folded_flux"][:, 0]
                    
                    fig_fold = plot_folded_lc(
                        phases, fluxes, selected_tic, 
                        period=params["period_days"], 
                        depth_ppm=params["depth_ppm"], 
                        duration_hrs=params["duration_hrs"]
                    )
                    st.plotly_chart(fig_fold, use_container_width=True)
                    
                # Row 3: Detail Cards
                st.markdown("---")
                dcol1, dcol2 = st.columns(2)
                
                with dcol1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Astrophysical Interpretation")
                    
                    if params["class"] == "transit":
                        st.write("""
                            **Transit Signal Confirmed**: The folded light curve shows a flat-bottomed, symmetric U-shape characteristic
                            of a planet passing in front of a star. The low orbital period and shallow depth (under 3,000 ppm)
                            indicate a small candidate (Super-Earth or Neptunian class) orbiting close to the host star.
                        """)
                    elif params["class"] == "eclipse":
                        st.write("""
                            **Eclipsing Binary (EB) Detected**: The binned folded curve shows a V-shape profile or deep dip,
                            typically representing a secondary star eclipsing the primary star. The transit depth is high (over 10,000 ppm),
                            which is physically incompatible with a planetary radius unless the host is a white dwarf.
                        """)
                    elif params["class"] == "blend":
                        st.write("""
                            **Blended/Background Binary (BEB)**: The signal has characteristics of a transit but is likely diluted
                            or blended with light from a nearby star, or presents a grazing binary configuration. The autoencoder reconstruction
                            error and classifier confidence indicate potential transit shape distortion.
                        """)
                    else:
                        st.write("""
                            **Other Variable / Stellar Activity**: The periodic signal detected is likely due to stellar spots, rotation,
                            pulsation, or residual instrument systematics. The signal lack the characteristic box/transit profile.
                        """)
                    
                    st.markdown(f"**Anomaly Score (Autoencoder reconstruction MSE):** `{params['anomaly_score']:.6f}`")
                    st.write("A higher reconstruction error indicates that the folded profile deviates significantly from the typical shapes seen during training.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with dcol2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Candidate Periodogram Peaks (Top-3)")
                    
                    # Print table of top candidates
                    table_data = []
                    for idx, cand in enumerate(candidates):
                        table_data.append({
                            "Rank": idx + 1,
                            "Period (days)": f"{cand['period']:.4f}",
                            "Epoch (BJD)": f"{cand['transit_time']:.4f}",
                            "Depth (ppm)": f"{int(cand['depth']*1e6):,}",
                            "SNR": f"{cand['snr']:.2f}",
                            "Flagged": "✅ Yes" if cand['snr'] >= snr_threshold else "❌ No"
                        })
                    st.table(table_data)
                    st.markdown('</div>', unsafe_allow_html=True)
