# ECG Anomaly Detection using Unsupervised Learning

This project implements an unsupervised anomaly detection system for ECG (Electrocardiogram) signals using various algorithms. The system analyzes ECG data to detect cardiac anomalies without using labeled data during training, though labels are used for evaluation and error analysis.

## Group Members
1. Dursun cihan Karadoğan
2. Minh chau Nguyen
3. Priyanka Sirikonda

## Project Overview

### Use Case
This project focuses on detecting cardiac anomalies in ECG signals, which is crucial for:
- Early detection of heart conditions
- Real-time monitoring of cardiac health
- Reducing false alarms in cardiac monitoring systems
- Improving automated ECG analysis systems

### Dataset
The project uses the MIT-BIH Arrhythmia Database, which contains ECG recordings with different types of heartbeats:
- Source: [MIT-BIH Arrhythmia Database](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- File: mitbih_train.csv
- Format: CSV
- Size: ~100MB
- 187 samples per beat
- 5 classes: Normal (N), Supraventricular (S), Ventricular (V), Fusion (F), Unknown (Q)

### Data Quality
- Sampling rate: 360 Hz
- Resolution: 11-bit
- Signal-to-noise ratio: >20dB
- Baseline wandering: <0.1mV
- Muscle noise: <0.05mV

### Dataset Statistics
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

## Project Structure

```
ecg_anomaly_detection/
├── data/               # Data directory
├── models/            # Model implementations
│   ├── fourier_model.py
│   ├── heuristic_model.py
│   └── var_model.py
├── notebooks/         # Jupyter notebooks for analysis
├── src/              # Source code
│   ├── data/         # Data processing modules
│   │   └── preprocess.py
│   └── training/     # Training scripts
│       └── train.py
├── tests/            # Unit tests
├── .gitignore        # Git ignore rules
├── README.md         # Project documentation
└── requirements.txt  # Project dependencies
```

## Algorithms

The project implements three different approaches for anomaly detection:

1. **Mean Absolute Deviation (MAD)**
   - Type: Heuristic-based
   - Approach: Uses statistical measures to detect deviations from normal patterns
   - Advantages: Simple, interpretable, works well with time series data
   - Implementation: Calculates MAD for each beat and compares with threshold

2. **Fourier Transform**
   - Type: Signal Processing
   - Approach: Analyzes frequency components of the ECG signals
   - Advantages: Can detect anomalies in frequency domain
   - Implementation: Transforms signals to frequency domain and analyzes deviations

3. **Vector Autoregression (VAR)**
   - Type: Machine Learning
   - Approach: Models temporal dependencies between different points in the time series
   - Advantages: Captures complex temporal patterns
   - Implementation: Models relationships between consecutive samples

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preprocessing:
```bash
python src/data/preprocess.py
```

2. Model Training:
```bash
python src/training/train.py
```

## Model Error Analysis

For each model, we analyze:
1. False Positives:
   - Why normal beats were classified as anomalies
   - Impact of noise and signal quality
   - Effect of baseline wandering

2. False Negatives:
   - Why anomalous beats were missed
   - Characteristics of undetected anomalies
   - Model limitations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 