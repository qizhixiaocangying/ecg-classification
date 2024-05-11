# ECG Signal Classification Project

## Overview

This project focuses on the classification of electrocardiogram (ECG) signals, aiming to accurately categorize heartbeats according to predefined classes. The workflow encompasses three core stages: preprocessing, feature extraction, and classification, with an emphasis on comprehensive analysis and selection of the optimal classifier for predictive tasks.

## Dataset and Preprocessing

The dataset utilized in this project consists of diverse ECG signals, which were initially processed to ensure high-quality analysis. Preprocessing steps involved:

- **Noise Reduction**: Implemented a 60Hz notch filter to remove powerline interference, followed by median filtering to eliminate transient spikes and baseline wander. Wavelet thresholding was also applied as a non-linear technique to further suppress noise while preserving signal integrity.
  
- **AAMI Standard Classification**: Applied the Association for the Advancement of Medical Instrumentation (AAMI) guidelines for heartbeat classification, ensuring conformity with clinical standards.

## Feature Extraction Details

### Time-Domain Features
- **R-Peak Metrics**: Extracted features around the R-peak of the QRS complex, including:
  - Pre-R interval
  - Post-R interval
  - Local R morphology characteristics
  - Global R morphology features, all normalized for consistency and comparability across different signals.

### Frequency-Domain Features
- **Spectral Analysis**: Comprehensive examination of frequency components comprising:
  - Frequency-domain features such as mean, standard deviation, skewness, and kurtosis of the power spectrum.
  - Spectral centroid (frequency expectation), reflecting the center of mass in the frequency distribution.
  - Frequency bandwidth measures capturing spectral spread.

### Time-Frequency Domain Features
- **Wavelet Transform**: Employed db1 wavelet at level 3, yielding 25 features from approximation coefficients, capturing both temporal and spectral variations.
- **Short-Time Fourier Transform (STFT)**: Calculated the mean and standard deviation of spectral contents within each time window, offering dynamic frequency-domain insights.

### Decomposition Domain
- **Empirical Mode Decomposition (EEMD)**: Derived from the first four Intrinsic Mode Functions (IMFs), extracting both time-domain and Welch spectrum-based frequency features, adhering to established literature methodologies.
- **Singular Value Decomposition (SVD)**: After reconstructing phase space with an embedding dimension of 5, SVD was used to isolate and analyze singular values, enhancing the understanding of signal structure.

### Nonlinear Features
- **Entropy Measures**: Estimated complexity and irregularity via approximate entropy, sample entropy, and fuzzy entropy, providing insights into signal dynamics.

### Macroscopic Features
- **Demographics**: Incorporated patient demographics – age and gender – as contextual factors, recognizing their potential influence on ECG patterns.

These meticulously selected features collectively encapsulate a broad spectrum of ECG signal attributes, enhancing the model's capacity to discern between different cardiac states effectively.

## Classifier Selection and Performance

To classify the ECG signals, five classifiers were trained and evaluated:

- **Support Vector Machine (SVM)**: Known for its robustness in high-dimensional spaces.
- **Random Forest (RF)**: Offers ensemble learning through decision trees for improved accuracy and interpretability.
- **Logistic Regression (LR)**: A fundamental algorithm for binary classification tasks.
- **AdaBoost**: Combines weak learners to form a strong classifier.
- **k-Nearest Neighbors (k-NN)**: Utilizes similarity measures for classification.

After rigorous testing and validation, **Support Vector Machine (SVM)** emerged as the optimal choice due to its superior performance metrics across various evaluation criteria. SVM's ability to handle complex, nonlinear data boundaries made it particularly suitable for the nuanced task of ECG signal classification.

## Contributions

This project contributes to the field of cardiovascular disease diagnosis by providing a robust framework for ECG signal analysis, potentially facilitating early detection and intervention strategies.
