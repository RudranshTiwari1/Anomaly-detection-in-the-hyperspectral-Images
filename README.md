# 🌈 Hyperspectral Image Anomaly Detection

This project focuses on **anomaly detection in hyperspectral images** using both deep learning (Autoencoder) and classical methods (RX Anomaly Detector). After comparison, the **RX Detector alone gave the best results**.

---

## 📌 Project Summary

- Input: Indian Pines Hyperspectral Image (`.mat` format)
- Methods Used:
  - PCA + Autoencoder → (`AnamolyMap.py`)
  - RX Anomaly Detector → (`_op.ipynb`)
- Output:
  - Anomaly Maps
  - Spectral Plots
  - ROC/AUC-based Evaluation (with Ground Truth)

---

## 🗂️ Files Description

| File Name              | Description                                       |
|------------------------|---------------------------------------------------|
| `AnamolyMap.py`        | PCA + Autoencoder for anomaly detection          |
| `_op.ipynb`            | RX Anomaly Detector (final output)               |
| `Gound_Truth.py`       | Loads and processes ground truth labels          |
| `ViewImage.py`         | Visualization utilities                          |
| `Indian_pines_corrected.mat` | Hyperspectral image data                     |
| `Indian_pines_gt.mat`  | Ground truth labels for Indian Pines             |

---

## ⚙️ How to Run

### Step 1: Clone the Repo

```bash
git clone https://github.com/RudranshTiwari1/Hyperspectral-Anomaly-Detection.git
cd Hyperspectral-Anomaly-Detection
```

### Step 2: Install Dependencies
```bash
pip install numpy matplotlib scipy scikit-learn tensorflow
```

### Step 3: Run Autoencoder
```bash
python AnamolyMap.py
```

### Step 4: Run RX Anomaly Detector
Open _op.ipynb in Jupyter Notebook and run all cells.
