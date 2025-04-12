# Electromyography (EMG) Analysis Project

A signal processing and classification pipeline for analyzing forearm surface EMG (sEMG) data to identify hand movements using time and frequency domain features and KNN classification.

## 📜 Project Overview

This project, developed for the **Digital Signal Processing** course at The American University in Cairo, processes EMG signals from three subjects performing hand gestures. The primary objectives were to:

- Preprocess raw EMG signals using high-pass filtering.
- Extract relevant features from time and frequency domains.
- Train and evaluate a K-Nearest Neighbors (KNN) classifier using leave-one-trial-out cross-validation.
- Compare segmentation strategies and fusion of features across multiple channels.

## 📁 Dataset Description

Each subject's data consists of:
- sEMG signals from electrodes placed around the forearm.
- A stimulus signal indicating the timing and type of hand movements (abduction and fist).
- A sampling frequency of 100 Hz.

## 🧠 Methods Used

### 1. Preprocessing
- **High-Pass Filter:** Butterworth filter with a 10 Hz cutoff frequency to remove low-frequency noise.

### 2. Feature Extraction

#### Time-Domain (Method 1)
- Segmented signals based on stimulus changes.
- Flattened each trial into 1D vectors.

#### Frequency-Domain (Method 2)
- Applied FFT and retained the first half (Nyquist).
- Used magnitude spectrum and flattened the result.

#### Combined Features (Method 3)
- Concatenated time and/or frequency features from all channels.

### 3. Classification

- **Classifier:** K-Nearest Neighbors (KNN)
- **Evaluation Strategy:** Leave-One-Trial-Out Cross Validation
- **Feature Scaling:** Standardization before classification

## 📊 Results Summary

- Frequency domain features consistently outperformed time domain features.
- Best accuracy per subject (fixed-length approach):
  - Subject 1: 95% (Channel 1, K=1)
  - Subject 2: 95% (Channel 2, K=9)
  - Subject 3: **100%** (Channel 3, K=1)
- Combining all channels:
  - Frequency-domain fusion yielded strong results.
  - Time-domain fusion reduced accuracy.

## 📈 Sample Visuals

Plots include:
- Time-domain and frequency-domain signals (before and after filtering)
- Accuracy vs. K for various channels
- Accuracy comparison between fusion approaches

## 📦 Tech Stack

- Python
- `numpy`, `scipy`, `matplotlib`, `sklearn`

## 📂 File Structure

```bash
📁 emg-analysis/
├── data/                  # Raw and preprocessed EMG datasets
├── plots/                 # Visualizations of signal processing and accuracy
├── src/                   # Main Python scripts
│   ├── preprocessing.py   # High-pass filter
│   ├── feature_extraction.py  # Time and frequency domain features
│   ├── classification.py  # KNN and evaluation
├── results/               # Accuracy metrics and results
└── README.md
