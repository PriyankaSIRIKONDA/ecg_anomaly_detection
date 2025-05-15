# Create README.md
echo "# ECG Anomaly Detection using Unsupervised Learning

## Group Members
1. Dursun cihan Karadoğan
2. Minh chau Nguyen
3. Priyanka Sirikonda

## Dataset
- Source: MIT-BIH Arrhythmia Database
- Link: https://www.kaggle.com/datasets/shayanfazeli/heartbeat
- File: mitbih_train.csv

## Use Case
This project implements unsupervised learning for anomaly detection in ECG (Electrocardiogram) signals. The goal is to identify unusual patterns or anomalies in heartbeats without using labeled data, which is crucial for early detection of cardiac abnormalities.

## Dataset Description
The dataset contains ECG recordings with different types of heartbeats:
- 187 samples per beat
- 5 classes: 
  - Normal (N)
  - Supraventricular (S)
  - Ventricular (V)
  - Fusion (F)
  - Unknown (Q)
- Each recording represents a single heartbeat
- Anomalies represent different types of abnormal heartbeats

## Dataset Statistics
- Total beats: 87,553
- Number of features: 187 (time series points)
- Number of classes: 5
- Normal beats: ~92%
- Anomaly rate: ~8%
- Distribution of anomaly types:
  - Supraventricular: ~2%
  - Ventricular: ~3%
  - Fusion: ~1%
  - Unknown: ~2%

## Proposed Algorithms
We will implement three different unsupervised learning approaches for anomaly detection:

1. **Mean Absolute Deviation (MAD)**
   - Type: Heuristic-based
   - Approach: Uses statistical measures to detect deviations from normal patterns
   - Advantages: Simple, interpretable, works well with time series data
   - Implementation: Will calculate MAD for each beat and compare with threshold

2. **Fourier Transform**
   - Type: Signal Processing
   - Approach: Analyzes frequency components of the ECG signals
   - Advantages: Can detect anomalies in frequency domain
   - Implementation: Will transform signals to frequency domain and analyze deviations

3. **Vector Autoregression (VAR)**
   - Type: Machine Learning
   - Approach: Models temporal dependencies between different points in the time series
   - Advantages: Captures complex temporal patterns
   - Implementation: Will model relationships between consecutive samples
