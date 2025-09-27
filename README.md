### Singer Classification (Traditional ML)

Train a traditional ML model (k-NN, SVM, RandomForest) to classify tracks by singer using hand-crafted audio features (MFCCs, chroma, spectral stats, tonnetz).

### Dataset Layout

Place your dataset like:

- `data_dir/singer_1/track1.wav`
- `data_dir/singer_1/track2.mp3`
- `data_dir/singer_2/track3.flac`
- ...

Any subdirectory under `data_dir` is treated as a class label.

### Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Train

```bash
python train_singer_classifier.py /absolute/path/to/data \
  --model all \
  --duration 30 \
  --grid \
  --output-dir /workspace/outputs
```

Common flags:
- **--model**: one of `knn`, `svm`, `rf`, or `all` (default: `all`)
- **--duration**: seconds of audio to load per track (default 30). Set to `0` to use full track.
- **--grid**: run a small GridSearchCV for the chosen model(s) for better accuracy.
- **--test-size**: test split fraction (default 0.2).
- **--sr**: sample rate for loading audio (default 22050).
- **--n-jobs**: parallelism for training/grid search (default -1).
- **--save-features**: write extracted features to CSV in the output directory.

Artifacts written to `--output-dir`:
- `model.joblib`, `scaler.joblib`, `label_encoder.joblib`
- `metrics.json`
- `features.csv` (if `--save-features`)

### Notes
- Features are time-aggregated statistics of MFCCs, chroma, spectral contrast, tonnetz, centroid, bandwidth, rolloff, ZCR, and RMS.
- Labels are derived from subdirectory names.
- The script prints accuracy and macro-F1; `metrics.json` includes a confusion matrix and per-class metrics.