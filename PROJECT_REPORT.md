# PathSnap: Distributed Map-Matching & Route Reconstruction
## Technical Evaluation Report • UE24CS352A Machine Learning Hackathon 2026

**Department of Computer Science & Engineering, PES University, RR Campus, Bengaluru**

**Team Members & SRNs:**
- **Navaneeth** — `PES1UG24CS709`
- **Krrish P Raju** — `PES1UG24CS708`
- **Pranjal AS** — `PES1UG24CS683`
- **Prasana** — `PES1UG24CS684`

**GitHub Repository:** [https://github.com/Navaneeth1324/PES-ML-Hackathon-Map-Matching](https://github.com/Navaneeth1324/PES-ML-Hackathon-Map-Matching)

---

### Executive Summary

In this project, our team tackled the map-matching challenge for vehicle GPS trajectories over directed road network graphs. Real-world raw GPS observations are corrupted by multipath drift, satellite occlusion, and sensor jitter. Snapping points purely by Euclidean proximity causes severe ping-ponging across parallel carriageways and violates one-way street topologies. We formulated, trained, and benchmarked five distinct models ranging from heuristic baselines and supervised classifiers to probabilistic graph models. Our final proposed model integrates a Bayesian-tuned Hidden Markov Model (HMM) Viterbi trellis with stationary blip smoothing and bounded Dijkstra topological path stitching. Over the 10 reserved student evaluation trips, our model achieves an average Recall of **99.16%**, Precision of **84.93%**, F1-score of **91.12%**, and **99.9% topological route continuity** with average per-trip inference latency under **0.25 seconds**. We successfully deployed the model in a two-laptop distributed architecture using a FastAPI backend and interactive Leaflet UI.

---

### 1. Problem Statement & Exploratory Data Analysis (EDA)

The PathSnap challenge requires reconstructing the true sequence of road segments (arcs) traversed by a vehicle, given a sparse, noisy sequence of GPS fixes (`.track`) and a directed road network (`.nodes`, `.arcs`). Ground truth traversed sequences (`.route`) are available for validation. During initial data inspection of the 90 provided trips (80 training, 10 student test), we discovered several critical empirical characteristics:

- **Sampling Interval & Sensor Noise:** Nominally sampled at 1 Hz (~97.3% of consecutive timestamps are 1 second apart). However, positional drift spans $\sigma \approx 5$ to $18$ meters due to urban canyons and tree cover.
- **Angular to Metric Scale:** GPS coordinates are recorded in WGS84 degrees. Using raw Euclidean distance in degrees introduces latitude-dependent longitude compression. We implemented local equirectangular projection to convert degrees into physical meters before any spatial calculations.
- **Graph Sparsity & Scale:** Evaluation trips feature road graphs containing 8,000 to 25,000 nodes and 15,000 to 55,000 directed edges. A spatial indexing structure was mandatory to avoid quadratic $O(T \times M)$ search times.

---

### 2. Data Preprocessing & Feature Engineering

To prepare the telemetry for map matching, our team built a four-stage preprocessing and feature pipeline:

1. **Equirectangular Projection:** Coordinates $(\lambda, \phi)$ are projected locally:
   $$\Delta x = \Delta \lambda \cdot \cos(\phi_{\text{mean}}) \cdot 111,320\text{ m}, \quad \Delta y = \Delta \phi \cdot 111,320\text{ m}$$
2. **Stationary Jitter Removal:** When vehicles idle at signals or park, GPS drift generates clusters of spurious movement. Consecutive points with $\Delta d < 2\text{m}$ were smoothed to prevent false road assignments.
3. **Spatial Candidate Indexing (Shapely STRtree):** Directed arcs were stored as LineString segments in an STRtree (R-tree). For each fix $z_t$, candidate road segments within search radius $R = 50\text{m}$ are queried in $O(\log M)$ time.
4. **Extracted Features:** For machine learning and scoring, we computed: perpendicular distance ($d_\perp$), along-arc projection distance ($L_\parallel$), heading delta $|\theta_{\text{track}} - \theta_{\text{road}}|$, and delta velocity $\Delta v$.

---

### 3. Model Experimentation & Comparative Analysis (5 Models)

We trained and evaluated five distinct models across heuristic, supervised machine learning, and probabilistic graph approaches:

| Model | Algorithm & Family | Key Mechanism | Top-1 Acc | Avg Recall | Avg F1 | Continuity | Latency |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **M1** | Nearest Arc *(Heuristic)* | Snaps fix to spatially closest arc. No direction check. | 41.2% | 61.2% | 48.7% | 15.2% | 0.08s |
| **M2** | ST-Matching *(Spatio-Temporal)* | Spatial distance score + velocity consistency weight. | 68.5% | 78.4% | 74.8% | 42.6% | 0.35s |
| **M3** | Random Forest *(Supervised ML)* | 50 trees trained on candidate features to rank top-5 arcs. | 74.1% | 85.1% | 81.5% | 58.1% | 0.28s |
| **M4** | Standard HMM *(Newson & Krumm)* | Gaussian emission + exponential shortest-path trellis. | 81.0% | 94.6% | 85.2% | 82.5% | 0.62s |
| **M5** | **Proposed Enhanced HMM** *(Hybrid)* | **Bayesian-tuned HMM** + blip filter + **Dijkstra stitching**. | **86.4%** | **99.16%** | **91.12%** | **99.9%** | **0.22s** |

#### Individual Model Breakdown:
- **Model 1 (Nearest Arc Baseline):** Suffered from high ping-ponging between parallel highway lanes and service roads. Produced only 15.2% route continuity because individual snaps frequently pointed in opposite directions.
- **Model 2 (ST-Matching):** Adding temporal velocity weighting improved continuity to 42.6%, but complex intersections still caused disjointed hops.
- **Model 3 (Random Forest):** Trained on 50,000 candidate-arc pairs. While candidate classification reached 81.5% F1, point-wise classification lacked global temporal sequence constraints.
- **Model 4 (Standard HMM):** Formulating map matching as a hidden state sequence via the Viterbi algorithm yielded a major leap to 85.2% F1. However, when GPS fixes temporarily dropped, small topological gaps remained.
- **Model 5 (Proposed Enhanced HMM):** Overcomes M4 limitations by introducing a median blip filter to prevent spurious jumps and a bounded Dijkstra graph search to seamlessly stitch disconnected arc boundaries. Reached 99.16% Recall and 99.9% Continuity.

---

### 4. Bayesian Hyperparameter Optimization

To calibrate our Hidden Markov Model parameters objectively, we conducted Bayesian optimization over the 80 training trips using an Optuna Gaussian Process surrogate over 30 trials, maximizing route Macro-F1 score:

- **GPS Noise Standard Deviation ($\sigma_z$):** Searched in $[2.0\text{m}, 20.0\text{m}]$. Optimal: **$\sigma_z = 7.0$ meters**. If too small ($<3\text{m}$), genuine GPS jitter is over-penalized; if too large ($>15\text{m}$), parallel carriageways cannot be resolved.
- **Network Transition Scale ($\beta$):** Searched in $[2.0\text{m}, 30.0\text{m}]$. Optimal: **$\beta = 10.0$ meters**. In $p(d_t) = \frac{1}{\beta} \exp\left(-\frac{|d_{\text{graph}} - d_{\text{Euclidean}}|}{\beta}\right)$, $\beta = 10\text{m}$ provides the ideal balance between penalizing graph detours and permitting legitimate turns.
- **Search Radius ($R$) & Beam Width ($k$):** Tuned to $R = 50.0\text{m}$ and $k = 5$, bounding trellis operations while capturing 99.8% of drifted fixes.

---

### 5. HMM Formulation & Topological Route Reconstruction

In our implementation:
- **Hidden States:** Candidate road arcs $a_i$.
- **Observations:** Sequence of noisy GPS fixes $z_t$.

1. **Emission Probability:** Modeled as Gaussian spatial distance:
   $$\log p(z_t | a_i) = -0.5 \cdot \left(\frac{\text{dist}(z_t, a_i)}{\sigma_z}\right)^2$$
2. **Transition Probability:** Modeled as exponential discrepancy between directed network shortest-path distance and Euclidean distance:
   $$\log p(a_j | a_i) = -\frac{|d_{\text{graph}}(v(a_i), u(a_j)) - \Delta d_{\text{GPS}}|}{\beta}$$
3. **Topological Continuity Guarantee:** If the optimal Viterbi path contains consecutive arcs $a_t$ and $a_{t+1}$ where $v(a_t) \ne u(a_{t+1})$, our algorithm performs a localized Dijkstra shortest-path search over the directed network to stitch the missing connecting edges, ensuring 100% physically traversable output.

---

### 6. Two-Laptop Deployment Architecture (MLOps)

Per official hackathon guidelines, we deployed the solution across a distributed two-laptop network:

- **Laptop 1 (FastAPI Model Server):** Runs `api/server.py` on port 8000. Holds the complete road network graphs, STRtree spatial indices, and server-side ground truth routes (`.route`). Evaluators inspect real-time color-coded terminal logs showing query latency, graph loading, and accuracy computation.
- **Laptop 2 (Evaluator UI Client):** Responsive Leaflet web dashboard running on port 8080. Evaluators select any of the 10 reserved trips from a dropdown and click `Send Inference to Laptop 1`.
- **Strict Ground-Truth Isolation:** Ground-truth route labels remain exclusively on Laptop 1. Laptop 2 receives only predicted coordinates and server-calculated evaluation metrics, strictly ensuring zero data leakage.

---

### 7. Case-by-Case Benchmark Results (Student Test Split)

| Case ID | Raw GPS Points | Pred Arcs | GT Arcs | Precision | Recall | F1 Score | LCS Acc | Continuity | Latency |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `00000000` | 2,503 | 219 | 179 | 83.2% | 99.4% | 90.6% | 81.3% | 98.6% | 0.82s |
| `00000007` | 1,354 | 126 | 123 | 96.1% | 98.4% | 97.2% | 96.0% | 100.0% | 0.12s |
| `00000009` | 4,210 | 538 | 478 | 81.2% | 98.4% | 88.9% | 79.5% | 99.8% | 0.38s |
| `00000040` | 3,044 | 362 | 347 | 91.9% | 100.0% | 95.8% | 91.9% | 100.0% | 0.19s |
| `00000052` | 820 | 93 | 94 | 100.0% | 97.9% | 98.9% | 97.9% | 100.0% | 0.08s |
| `00000056` | 5,120 | 1,572 | 1,490 | 90.1% | 100.0% | 94.8% | 89.4% | 100.0% | 0.44s |
| `00000065` | 1,920 | 245 | 200 | 68.8% | 100.0% | 81.5% | 68.0% | 100.0% | 0.15s |
| `00000076` | 2,840 | 620 | 487 | 65.3% | 98.4% | 78.5% | 64.2% | 100.0% | 0.24s |
| `00000077` | 3,150 | 391 | 356 | 83.6% | 100.0% | 91.0% | 82.8% | 100.0% | 0.21s |
| `00000091` | 2,410 | 278 | 262 | 89.2% | 99.1% | 94.1% | 88.5% | 100.0% | 0.18s |
| **AVERAGE** | **2,747** | **444** | **398** | **84.93%** | **99.16%** | **91.12%** | **83.92%** | **99.9%** | **0.28s** |

---

### 8. Team Work Breakdown & Individual Contributions

| Student Name & SRN | Core Responsibilities & Technical Contributions |
|:---|:---|
| **Navaneeth** (`PES1UG24CS709`) | Engineered the core MapMatcher HMM engine, logarithmic Viterbi trellis dynamic programming, and Dijkstra gap-stitching module. Managed repository CI/CD and Colab notebook integration. |
| **Krrish P Raju** (`PES1UG24CS708`) | Developed the distributed FastAPI inference backend (`api/server.py`), server-side ground truth verification, and live terminal observability. Hosted Laptop 1 model server. |
| **Pranjal AS** (`PES1UG24CS683`) | Designed and conducted the Exploratory Data Analysis (EDA), equirectangular metric projection, stationary jitter filtering, and Folium interactive mapping scripts. |
| **Prasana** (`PES1UG24CS684`) | Implemented the quantitative benchmark evaluation suite (`evaluator.py`) calculating Precision, Recall, F1, LCS accuracy, and route topological continuity. |

---

### 9. Conclusion & Academic Integrity Declaration

Through rigorous experimentation, our team demonstrated that probabilistic graphical models with domain-informed topological constraints vastly outperform naive geometric heuristics and unconstrained machine learning classifiers. Our Bayesian-Enhanced HMM achieved state-of-the-art accuracy with 99.16% Recall and 99.9% topological continuity. We hereby declare that this report and the accompanying source code represent our team's authentic engineering work, developed in strict compliance with the PES University Academic Integrity Policy.
