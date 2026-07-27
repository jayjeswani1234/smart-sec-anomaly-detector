<<<<<<< HEAD
# smart-sec-anomaly-detector
=======
# AI-Powered Behavioral Anomaly Detection for Cybersecurity

A compact, end-to-end prototype for the SIH idea:
- Generate synthetic access logs with realistic normal and anomalous behavior
- Train an anomaly detector on normal activity
- Classify attack type for flagged events
- Explain alerts for SOC analysts
- Explore results in a Streamlit dashboard

## Features
- Synthetic schema aligned with the problem statement
- Supports anomaly patterns such as:
  - brute force
  - impossible travel
  - credential stuffing
  - lateral movement
  - device spoofing
  - low-and-slow
  - insider drift
- Cold-start friendly baseline profiles
- Top-risk alert ranking for analyst review
- Explainable reasons for each alert

## Project structure
```
cyber_anomaly_detection/
├── app.py
├── train.py
├── infer.py
├── requirements.txt
├── src/
│   ├── config.py
│   ├── synthetic_data.py
│   ├── features.py
│   ├── model.py
│   ├── explain.py
│   └── utils.py
├── data/
└── models/
```

## Quick start

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Outputs
Training will save:
- `models/anomaly_detector.joblib`
- `models/attack_classifier.joblib`
- `models/metadata.json`

## Notes
This prototype is designed for the SIH presentation and demo workflow. It is not a production security product.
>>>>>>> c0c643e (Add Docker configuration and Streamlit setup)
