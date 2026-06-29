# Wind Tunnel Testing: Task 2 Wake Measurement Analysis (Group 2)

## 1. Introduction and Physics Background
This repository processes high-frequency ($10 \text{ kHz}$) time-series measurement data from a wind turbine model tested in a wind tunnel. The test matrix covers three yaw conditions: $-30^\circ$, $0^\circ$, and $+30^\circ$. 

### The "Steady State" Dilemma in Wake Aerodynamics
Wind turbine wakes are inherently unsteady. Due to the rotating blades and hub vortices, the velocity field behind the rotor cannot be treated as a simple steady-state flow. The flow exhibits **cyclostationarity**—meaning the statistical properties of the turbulence and velocity deficit fluctuate periodically with the passing of the blades. Therefore, averaging $100,000$ data points into a single "Mean" masks the true physical properties of the wake.

## 2. Data Processing Pipeline Architecture

To transition from 100,000 raw samples per point to a scientifically representative 2D Wake Map, we employ a 3-Phase automated data extraction pipeline:

### Phase 1: Spectral Diagnostics (The Waterfall Heatmap)
* **Goal:** Understand the spatial distribution of energy in the frequency domain.
* **Method:** We use Welch’s method (`scipy.signal.welch`) to compute the Power Spectral Density (PSD) for each of the 23 measurement points. A 2D heatmap is plotted (Y-axis = Measurement Point, X-axis = Frequency). 
* **Physics:** This visualizes whether structural frequencies (like blade-pass frequency, $f_{BPF}$, or vortex shedding) are localized to the blade tips (outer measurement points) or the nacelle wake (inner measurement points).

### Phase 2: Feature Extraction and Decision Matrix
* **Goal:** Automatically decide which statistical metric best represents the flow at a given location.
* **Method:** A peak-finding algorithm locates the highest energy spike in the PSD. We establish a threshold for the `Peak_PSD` magnitude.
    * **Stochastic Flow:** If the PSD magnitude is below the threshold, the flow is treated as white-noise turbulence. The **Median** is selected to represent the bulk flow while ignoring transient outliers.
    * **Periodic Flow:** If the PSD magnitude exceeds the threshold, the signal is dominated by a periodic physical structure. The **Mean** (or a narrow-band filtered amplitude) is selected to capture the fundamental aerodynamic state.
* **Output:** A `.csv` Feature Table documenting the exact mathematical justification for every spatial point.

### Phase 3: Wake Map Generation
* **Goal:** Visualize the velocity deficit and wake steering effects.
* **Method:** We extract the physical coordinates (`[Y, Z]`) from the `.mat` arrays. Using the `Selected_Value` generated in Phase 2, we generate a `tricontourf` map to interpolate the flow field across the 23 points. 
* **Physics:** By comparing the Wake Maps of $-30^\circ, 0^\circ, \text{and } +30^\circ$, engineers can directly observe "wake deflection"—how yawing the turbine pushes the low-velocity wake away from the centerline, a critical concept in wind farm layout optimization.

## 3. Developer Notes
* **MATLAB Array Dimensions:** Data imported via `mat73` retains MATLAB's matrix dimensions (e.g., `(10000, 1)`). To prevent `ValueError: x must be a 1-D array` during `scipy` signal processing, `np.squeeze()` must be applied during the raw data extraction loop.
* **Memory Management:** The current `DataProcessor` caches the full FFT and Welch arrays in memory for all 23 points. For larger arrays, consider writing intermediate outputs to HDF5.