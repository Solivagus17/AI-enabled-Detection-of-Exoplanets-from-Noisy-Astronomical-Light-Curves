"""
Convert .keras models to .h5 format for cross-platform compatibility.
Run this locally: python pipeline/convert_models.py
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

clf_keras = os.path.join(MODELS_DIR, "clf_model.keras")
ae_keras  = os.path.join(MODELS_DIR, "ae_model.keras")
clf_h5    = os.path.join(MODELS_DIR, "clf_model.h5")
ae_h5     = os.path.join(MODELS_DIR, "ae_model.h5")

print("Loading clf_model.keras ...")
clf = tf.keras.models.load_model(clf_keras)
clf.save(clf_h5)
print(f"Saved → {clf_h5}")

print("Loading ae_model.keras ...")
ae = tf.keras.models.load_model(ae_keras)
ae.save(ae_h5)
print(f"Saved → {ae_h5}")

print("Done! Upload clf_model.h5 and ae_model.h5 to HF.")
