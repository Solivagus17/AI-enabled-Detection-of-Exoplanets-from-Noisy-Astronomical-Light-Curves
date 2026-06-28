# ExoHunter: A Deep Learning and Box Least Squares Pipeline for TESS Exoplanet Detection and Classification

**Author: Sarthakk Anjariya**  
**Astrophysics Machine Learning Research & Engineering**

---

## 1. Introduction and Astrophysical Context

Detecting exoplanets (planets orbiting stars outside our solar system) is one of the most exciting fields in modern astronomy. The NASA Transiting Exoplanet Survey Satellite (TESS) mission monitors bright stars across the entire sky to search for transiting planets.

### The Transit Method
When an exoplanet's orbit is aligned such that it passes in front of its host star relative to our line of sight, it blocks a small fraction of the star's light. This event, called a **transit**, causes a periodic dip in the star's observed brightness (flux) over time, resulting in a characteristic **light curve**.

```
    Flux
    1.0 ─────────────────┐             ┌───────────────── Out-of-transit baseline
                         │             │
                         │   Transit   │
                         └──────┬──────┘
                                │
    1.0 - Depth ────────────────┴──────────────────────── Transit dip bottom
```

### Signal Classification Challenges
Searching for transits is challenging because several astrophysical phenomena can mimic planet transits:
1. **Planetary Transits (transit)**: Flat-bottomed, symmetric, and relatively shallow (usually $< 2\%$ or $20,000$ ppm depth).
2. **Eclipsing Binaries (eclipse)**: A binary star system where one star eclipses another. These produce very deep, V-shaped dips, and often show alternating primary and secondary eclipses of different depths.
3. **Blended / Background Binaries (blend)**: The light curve is contaminated by a nearby eclipsing binary or background star, diluting the signal and making a deep eclipse look like a shallow planetary transit.
4. **Stellar Variability / Noise (other)**: Starspots, stellar rotation, pulsations, flares, or instrument systematics can create periodic trends and dips that mimic transits.

---

## 2. Ingestion and Preprocessing Pipeline

To ensure a robust, high-fidelity dataset for the machine learning models, the pipeline implements a rigorous ingestion and preprocessing workflow.

### Data Ingestion Strategy
We fetch TESS light curves using the `Lightkurve` python library, querying the Mikulski Archive for Space Telescopes (MAST) API.
- We restrict downloads to **short-cadence (2-minute)** light curves processed by the Science Processing Operations Center (SPOC).
- Short-cadence data is critical for resolving transit ingress and egress shapes, which are key features for distinguishing planets from eclipsing binaries.
- To prevent unnecessary API queries, FITS files are cached locally in `data/raw/` with a total disk usage cap of 300MB.

### Preprocessing Steps
Raw light curves contain stellar variability, instrument drifts, and cosmic ray outliers. The preprocess module cleans the light curves through the following sequential steps:

1. **Outlier Removal**: We apply standard 3-sigma clipping (`remove_outliers(sigma=3)`) to eliminate high-frequency spikes caused by cosmic ray hits on the detector.
2. **Variability Flattening**: We apply a Savitzky-Golay filter (`.flatten(window_length=401)`) to remove slow stellar rotation and spacecraft pointing drifts, flattening the out-of-transit baseline to a value of exactly 1.0.
3. **Gap Filling**: We interpolate over data gaps (caused by spacecraft data downlinks or momentum dumps) using linear interpolation (`.fill_gaps()`) to ensure a continuous time series.
4. **Normalization**: The flux is divided by its median value, establishing a standard baseline of 1.0.
5. **Fixed-Length Standardization**: Deep learning networks require a fixed input shape. We standardize the light curves to exactly **4000 points** by truncating longer series or padding shorter ones with a value of 1.0 (baseline flux).

---

## 3. Signal Detection and Phase Folding

The detection stage searches for periodic signals in the preprocessed light curve using the Box Least Squares (BLS) algorithm.

### Box Least Squares (BLS)
BLS fits a step-function box to the light curve, parameterized by:
- Period ($P$): The orbital period of the candidate planet.
- Transit Epoch ($t_0$): The midpoint time of the transit.
- Depth ($\delta$): The fraction of light blocked.
- Duration ($d$): The width of the transit box.

The power spectrum represents the likelihood of a transit at each period. We identify local maxima in the power spectrum using peak-finding algorithms to locate the **top 3 candidate periods**.

### Phase Folding and Binning
Once a candidate period $P$ and epoch $t_0$ are identified, we fold the light curve. The phase $\phi$ for each data point is calculated as:
\[\phi = \frac{(t - t_0 + 0.5P) \pmod P - 0.5P}{P}\]
This shifts the transit midpoint to $\phi = 0.0$, ranging the phase space from $-0.5$ to $+0.5$.

We bin the phase-folded light curve into **200 uniform bins** from phase $-0.5$ to $+0.5$. Binning reduces white noise by a factor of $\sqrt{N_{bin}}$, producing a high-SNR 1D profile of shape `(200, 1)` representing the transit shape, which is scale- and period-invariant.

### Signal-to-Noise Ratio (SNR)
The SNR of the transit is computed as:
\[\text{SNR} = \frac{\text{Transit Depth}}{\sigma_{\text{out-of-transit}}}\]
Where $\sigma_{\text{out-of-transit}}$ is the standard deviation of the binned flux in the out-of-transit phases (phases $\le -0.2$ and $\ge 0.2$). We flag detections exceeding the standard astrophysics threshold:
\[\text{SNR} > 7.1\]

---

## 4. Deep Learning Classification and Anomaly Detection

To classify the binned folded phase curves, we train two distinct neural network architectures in TensorFlow.

### Model A: CNN-LSTM Classifier
This network combines 1D Convolutional layers (to extract local shape features) with an LSTM layer (to model the sequence and temporal asymmetry).

```
   Folded Input (200, 1)
            │
    [Conv1D - 64 filters] -> ReLU
            │
       [MaxPool1D]
            │
    [Conv1D - 128 filters] -> ReLU
            │
       [MaxPool1D]
            │
        [LSTM - 64]
            │
     [Dense - 64] -> ReLU
            │
       [Dropout - 0.3]
            │
     [Dense - 4] -> Softmax
            │
   Class Probabilities: (Transit, Eclipse, Blend, Other)
```

### Data Augmentation
With only 100 TIC IDs, we implement data augmentation to expand the training set 10-fold (to 800 training samples) and prevent overfitting:
- **Phase Shifting (Phase Roll)**: We roll the binned curve horizontally by $\pm 5$ bins, simulating small epoch determination errors.
- **Gaussian Noise**: We inject Gaussian noise ($\sigma = 0.0005$) to simulate lower SNR instruments.
- **Depth Scaling**: We scale the transit dip depth by a random factor between $0.8$ and $1.2$.

### Model B: Autoencoder Anomaly Detector
An Autoencoder is trained to reconstruct the folded phase curve.
- **Encoder**: Compresses the `(200, 1)` input into a low-dimensional bottleneck representation of size `8` using 1D convolutions and a Dense layer.
- **Decoder**: Reconstructs the original `(200, 1)` curve using 1D Transposed Convolutions.
- **Anomaly Score**: Computed as the Mean Squared Error (MSE) between the input and the reconstructed output. An anomalous light curve (e.g. a weird systematic or rare transit shape) will reconstruct poorly, yielding a high anomaly score.

---

## 5. Parameter Estimation and Uncertainty Quantification

Once a transit is detected and classified, we estimate physical parameters and quantify their uncertainties using bootstrapping.

### Physical Parameters
- **Orbital Period ($P$)**: The peak BLS period.
- **Transit Depth ($\delta$)**: Establishes planetary size relative to the star:
  \[\delta = 1.0 - \min(\text{flux}_{\text{binned}})\]
- **Transit Duration ($d$)**: Computed as the width of the dip at the half-depth level ($1.0 - \delta/2.0$), converted to hours.
- **Confidence Score ($C$)**: Combined index of classification probability $P_{\text{transit}}$ and SNR:
  \[C = P_{\text{transit}} \times \left(1 - e^{-\text{SNR} / 7.1}\right)\]

### Bootstrap Uncertainty (1000x Resampling)
To calculate robust 1-sigma uncertainty bounds under non-Gaussian stellar noise, we perform **1000 bootstrap iterations**:
1. We resample the phase-folded raw light curve points with replacement.
2. We bin the resampled light curve into 200 bins.
3. We compute the transit depth and duration on the resampled binned curve.
4. We run a fast local BLS search (using a dense grid around the peak period) on 100 raw bootstrap resamples to find the period distribution.
5. The standard deviation of the bootstrapped parameter values is reported as the 1-sigma uncertainty error (`depth_err`, `duration_err`, `period_err`).

---

## 6. Conclusion and Dashboard

The final exoplanet pipeline is integrated into an interactive **Streamlit Dashboard** (`app.py`). The dashboard allows hackathon judges and users to:
1. Select from the 100 cached TIC IDs or upload a custom FITS file.
2. Visualize the raw light curve showing detrended flux and highlighted transit regions.
3. Compare the folded phase curve side-by-side with the binned median points and best-fit BLS box model.
4. Inspect real-time metric cards showing period, depth, duration, SNR, confidence, and anomaly scores.
5. Filter targets by category or adjust the SNR threshold slider to see how it affects detection sensitivity.
