# 🚗 AI-Based Driver Drowsiness Detection System

An AI-powered real-time Driver Drowsiness Detection System that monitors a driver's eye movement, yawning behavior, and facial landmarks to detect fatigue and alert the driver before accidents occur.

This project combines **Computer Vision**, **Deep Learning**, and **MediaPipe** to improve road safety through intelligent driver monitoring.

---

## 📌 Features

- 👁️ Eye Aspect Ratio (EAR) based eye closure detection
- 😮 Yawn detection using a CNN model
- 🧠 LSTM-based fatigue analysis
- 🎯 Real-time face landmark detection using MediaPipe
- 📷 Live webcam monitoring
- 🔔 Drowsiness alert with Arduino buzzer (optional — works without hardware too)
- ⚡ Fast real-time inference
- 📊 Fatigue score estimation

---

## 🛠️ Technologies Used

Python, OpenCV, TensorFlow / Keras, MediaPipe, NumPy, Pandas, Scikit-learn, Arduino (optional), Joblib

---

## 📂 Project Structure

```
ai-drowsiness-detection/
│
├── src/                      # application source (inference)
│   ├── main.py                # entry point
│   ├── config.py              # all thresholds & settings, read once, used everywhere
│   ├── features/
│   │   ├── eye.py
│   │   ├── mouth.py
│   │   └── pose.py
│   └── alerts/
│       └── arduino.py         # buzzer/serial logic, isolated from main loop
│
├── training/                 # scripts to build the models (not run in production)
│   ├── extract_features.py
│   ├── prepare_data.py
│   ├── train_lstm.py
│   └── train_yawn_model.py
│
├── models/                   # trained artifacts (gitignored — see below)
│   ├── drowsiness_model.h5
│   ├── yawn_model.h5
│   └── scaler.save
│
├── data/                     # datasets (gitignored — see below)
│   ├── dataset.csv
│   └── processed_data.csv
│
├── .env.example               # copy to .env to set your Arduino port
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/hemrajmanikandan/ai-drowsiness-detection.git
cd ai-drowsiness-detection

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Copy the environment template and fill in your Arduino port if you're using the buzzer:

```bash
cp .env.example .env
```

If you don't use `python-dotenv`, just export the variable directly:

```bash
export ARDUINO_PORT=COM5   # or /dev/ttyUSB0 on Linux
```

---

## ▶️ Running the Project

From the project root:

```bash
python -m src.main
```

The webcam will open automatically. Press `Esc` to quit.

---

## 🧠 Model Training

```bash
python training/extract_features.py
python training/prepare_data.py
python training/train_lstm.py
python training/train_yawn_model.py
```

Trained models are written to `models/`.

---

## 📊 Workflow

```
Webcam
  │
  ▼
MediaPipe Face Detection
  │
  ▼
Feature Extraction (EAR, MAR, Head Pose)
  │
  ▼
CNN Yawn Detection
  │
  ▼
LSTM Fatigue Prediction
  │
  ▼
Drowsiness Decision
  │
  ▼
Alarm & Warning
```

---

## 📁 Dataset

Raw datasets are **not included** in this repository due to size. Public options:

- Driver Drowsiness Dataset (DDD)
- Yawn Detection Dataset
- MediaPipe Face Mesh

Place them under:

```
data/dataset_raw/
data/yawn_dataset/
```

## 📦 Model & data artifacts

`models/*.h5`, `models/*.save`, and `data/*.csv` are gitignored by default — trained models and large CSVs shouldn't live directly in git history. For a real release, host them via **Git LFS**, a cloud bucket, or GitHub Releases, and document the download step here. If you'd rather keep committing them as-is for now, just remove those lines from `.gitignore`.

---

## 📌 Future Improvements

- Mobile application integration
- Night-time drowsiness detection
- Driver distraction detection
- Emotion recognition
- Cloud monitoring dashboard
- Edge AI deployment using Raspberry Pi
- Unit tests for feature extraction (`tests/`)

---

## 👨‍💻 Author

**Hemraj Manikandan** — Electronic and Instrumentation Engineering (EIE), SRM Institute of Science and Technology
GitHub: https://github.com/hemrajmanikandan

---

## 📄 License

This project is intended for educational and research purposes.
