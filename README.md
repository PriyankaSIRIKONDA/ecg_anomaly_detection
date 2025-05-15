# ECG Anomaly Detection using Unsupervised Learning

This project implements an unsupervised anomaly detection system for ECG (Electrocardiogram) signals using various algorithms. The system analyzes ECG data to detect cardiac anomalies without using labeled data during training, though labels are used for evaluation and error analysis.

## Group Members
1.Dursun cihan Karadoğan
2.Minh chau Nguyen
3.Priyanka Sirikonda


## Dataset
The project uses the MIT-BIH Arrhythmia Database, which contains 48 half-hour excerpts of two-channel ambulatory ECG recordings. The dataset is available at: [MIT-BIH Arrhythmia Database](https://www.physionet.org/content/mitdb/1.0.0/)

### Use Case
This project focuses on detecting cardiac anomalies in ECG signals, which is crucial for:
- Early detection of heart conditions
- Real-time monitoring of cardiac health
- Reducing false alarms in cardiac monitoring systems
- Improving automated ECG analysis systems

### Dataset Description
The MIT-BIH Arrhythmia Database contains:
- 48 half-hour excerpts of two-channel ambulatory ECG recordings
- 23 recordings chosen at random from a set of 4000 24-hour ambulatory ECG recordings
- 25 recordings selected to include less common but clinically significant phenomena
- Sampling rate: 360 Hz
- Resolution: 11-bit resolution over a 10 mV range
- Two channels: Modified limb lead II (MLII) and one of V1, V2, V4, or V5

### Anomaly Definition
In this context, anomalies represent:
- Premature ventricular contractions (PVCs)
- Premature atrial contractions (PACs)
- Bundle branch blocks
- Other cardiac arrhythmias
- Abnormal heart rhythms

### Dataset Statistics
- Total number of beats: ~110,000
- Number of different beat types: 15
- Percentage of normal beats: ~80%
- Percentage of anomalous beats: ~20%
- Distribution of anomalies:
  - Premature ventricular contractions (PVCs): ~7%
  - Premature atrial contractions (PACs): ~3%
  - Other anomalies: ~10%

## Project Structure

```
ecg_anomaly_detection/
├── data/               # Data directory
│   ├── mitbih_train.csv    # Training dataset
│   └── EDA.ipynb      # Exploratory Data Analysis notebook
├── models/            # Saved model checkpoints
├── src/              # Source code
│   ├── data/         # Data processing modules
│   ├── models/       # Model architecture definitions
│   │   ├── heuristic/    # Heuristic-based models
│   │   ├── ml/          # Machine learning models
│   │   └── evaluation/  # Model evaluation scripts
│   └── utils/        # Utility functions
├── tests/            # Unit tests
├── notebooks/        # Jupyter notebooks for analysis
└── requirements.txt  # Project dependencies
```

## Algorithms

The project implements three different approaches for anomaly detection:

1. Heuristic-based:
   - Mean Absolute Deviation (MAD)
   - Local/Global Statistical Analysis

2. Machine Learning-based:
   - Fourier Transform Analysis
   - Vector Autoregression

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
python src/models/train.py
```

3. Model Evaluation:
```bash
python src/models/evaluation/evaluate.py
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 