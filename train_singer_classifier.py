#!/usr/bin/env python3
"""
Train a traditional ML classifier (k-NN, SVM, RandomForest) to identify the singer of a track.

Expected dataset structure:
  data_dir/
    singer_1/track_a.wav
    singer_1/track_b.mp3
    singer_2/track_c.flac
    ...

This script will:
- Walk the dataset directory and extract robust audio features per track
- Split into train/test sets with stratification
- Scale features and train one or more models
- Evaluate on the test set and save the best model, scaler, and label encoder

Outputs:
- <output_dir>/model.joblib
- <output_dir>/scaler.joblib
- <output_dir>/label_encoder.joblib
- <output_dir>/metrics.json
- <output_dir>/features.csv (optional)
"""
import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
import librosa  # noqa: E402

try:
	from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
	def tqdm(iterable, **_kwargs):  # type: ignore
		return iterable


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


@dataclass
class FeatureConfig:
	sample_rate: int = 22050
	duration_s: Optional[float] = 30.0
	hop_length: int = 512
	n_mfcc: int = 20


def is_audio_file(path: str) -> bool:
	_, ext = os.path.splitext(path)
	return ext.lower() in SUPPORTED_EXTENSIONS


def list_audio_files_by_label(dataset_dir: str) -> List[Tuple[str, str]]:
	file_label_pairs: List[Tuple[str, str]] = []
	for label in sorted(os.listdir(dataset_dir)):
		label_dir = os.path.join(dataset_dir, label)
		if not os.path.isdir(label_dir):
			continue
		for root, _dirs, files in os.walk(label_dir):
			for fname in files:
				fpath = os.path.join(root, fname)
				if is_audio_file(fpath):
					file_label_pairs.append((fpath, label))
	return file_label_pairs


def load_audio_mono(path: str, sample_rate: int, duration_s: Optional[float]) -> Tuple[np.ndarray, int]:
	y, sr = librosa.load(path, sr=sample_rate, mono=True, duration=duration_s)
	y, _ = librosa.effects.trim(y, top_db=30)
	return y, sr


def summarize_feature_matrix(name: str, matrix: np.ndarray, stats: Tuple[str, ...] = ("mean", "std")) -> Dict[str, float]:
	if matrix.ndim == 1:
		matrix = matrix[np.newaxis, :]
	feature_dict: Dict[str, float] = {}
	for band_index in range(matrix.shape[0]):
		band_values = matrix[band_index]
		if "mean" in stats:
			feature_dict[f"{name}_{band_index}_mean"] = float(np.mean(band_values))
		if "std" in stats:
			feature_dict[f"{name}_{band_index}_std"] = float(np.std(band_values))
		if "skew" in stats:
			feature_dict[f"{name}_{band_index}_skew"] = float(pd.Series(band_values).skew())
		if "kurt" in stats:
			feature_dict[f"{name}_{band_index}_kurt"] = float(pd.Series(band_values).kurt())
	return feature_dict


def extract_features_from_audio(y: np.ndarray, sr: int, config: FeatureConfig) -> Dict[str, float]:
	stft = np.abs(librosa.stft(y=y, hop_length=config.hop_length))
	mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.n_mfcc, hop_length=config.hop_length)
	chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sr, hop_length=config.hop_length)
	spectral_centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)
	spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)
	spectral_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
	spectral_rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)
	zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y, hop_length=config.hop_length)
	rms = librosa.feature.rms(S=stft)
	y_harmonic = librosa.effects.harmonic(y)
	try:
		tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
	except Exception:
		tonnetz = np.zeros((6, max(1, mfcc.shape[1])))

	features: Dict[str, float] = {}
	features.update(summarize_feature_matrix("mfcc", mfcc, stats=("mean", "std")))
	features.update(summarize_feature_matrix("chroma", chroma_stft, stats=("mean", "std")))
	features.update(summarize_feature_matrix("contrast", spectral_contrast, stats=("mean", "std")))
	features.update(summarize_feature_matrix("tonnetz", tonnetz, stats=("mean", "std")))
	features.update(summarize_feature_matrix("centroid", spectral_centroid, stats=("mean", "std")))
	features.update(summarize_feature_matrix("bandwidth", spectral_bandwidth, stats=("mean", "std")))
	features.update(summarize_feature_matrix("rolloff", spectral_rolloff, stats=("mean", "std")))
	features.update(summarize_feature_matrix("zcr", zero_crossing_rate, stats=("mean", "std")))
	features.update(summarize_feature_matrix("rms", rms, stats=("mean", "std")))
	return features


def build_features_dataframe(
	dataset_dir: str,
	config: FeatureConfig,
	max_files_per_class: Optional[int] = None,
) -> pd.DataFrame:
	records: List[Dict[str, object]] = []
	pairs = list_audio_files_by_label(dataset_dir)
	if max_files_per_class is not None:
		class_counts: Dict[str, int] = {}
		filtered_pairs: List[Tuple[str, str]] = []
		for path, label in pairs:
			count = class_counts.get(label, 0)
			if count < max_files_per_class:
				filtered_pairs.append((path, label))
				class_counts[label] = count + 1
		pairs = filtered_pairs

	for path, label in tqdm(pairs, desc="Extracting features"):
		try:
			y, sr = load_audio_mono(path, config.sample_rate, config.duration_s)
			if y.size == 0:
				continue
			track_features = extract_features_from_audio(y, sr, config)
			track_features["label"] = label
			track_features["path"] = path
			records.append(track_features)
		except Exception as exc:
			print(f"Warning: failed to process {path}: {exc}")

	if not records:
		raise RuntimeError("No features extracted. Check dataset path and audio formats.")

	return pd.DataFrame.from_records(records)


def get_models(random_state: int, n_jobs: int) -> Dict[str, object]:
	models: Dict[str, object] = {
		"knn": KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski"),
		"svm": SVC(C=10.0, kernel="rbf", gamma="scale", probability=False, random_state=random_state),
		"rf": RandomForestClassifier(
			n_estimators=300,
			max_depth=None,
			random_state=random_state,
			n_jobs=n_jobs,
		),
	}
	return models


def get_param_grids() -> Dict[str, Dict[str, List[object]]]:
	return {
		"knn": {
			"n_neighbors": [3, 5, 7, 11],
			"weights": ["uniform", "distance"],
		},
		"svm": {
			"C": [1.0, 10.0, 50.0, 100.0],
			"gamma": ["scale", "auto"],
		},
		"rf": {
			"n_estimators": [200, 300, 500],
			"max_depth": [None, 20, 40],
		},
	}


def train_and_evaluate(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray,
	y_test: np.ndarray,
	model_key: str,
	run_grid_search: bool,
	random_state: int,
	n_jobs: int,
) -> Tuple[str, object, Dict[str, object]]:
	available_models = get_models(random_state, n_jobs)
	if model_key not in available_models:
		raise ValueError(f"Unknown model '{model_key}'")

	model = available_models[model_key]

	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

	if run_grid_search:
		param_grid = get_param_grids()[model_key]
		search = GridSearchCV(
			estimator=model,
			param_grid=param_grid,
			scoring="accuracy",
			cv=cv,
			n_jobs=n_jobs,
			verbose=1,
		)
		search.fit(X_train, y_train)
		best_model = search.best_estimator_
		best_params = search.best_params_
		cv_best_score = float(search.best_score_)
	else:
		best_model = model
		best_model.fit(X_train, y_train)
		best_params = getattr(best_model, "get_params", lambda: {})()
		cv_scores = metrics.cross_val_score(best_model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=n_jobs)
		cv_best_score = float(np.mean(cv_scores))

	y_pred = best_model.predict(X_test)
	accuracy = float(metrics.accuracy_score(y_test, y_pred))
	f1_macro = float(metrics.f1_score(y_test, y_pred, average="macro"))
	class_report = metrics.classification_report(y_test, y_pred, output_dict=True)
	conf_matrix = metrics.confusion_matrix(y_test, y_pred).tolist()

	metrics_dict: Dict[str, object] = {
		"model": model_key,
		"best_params": best_params,
		"cv_best_accuracy": cv_best_score,
		"test_accuracy": accuracy,
		"test_f1_macro": f1_macro,
		"classification_report": class_report,
		"confusion_matrix": conf_matrix,
	}
	return model_key, best_model, metrics_dict


def main() -> None:
	parser = argparse.ArgumentParser(description="Train singer classification with traditional ML")
	parser.add_argument("data_dir", type=str, help="Path to dataset root with subdirectories per singer")
	parser.add_argument("--model", type=str, default="all", choices=["knn", "svm", "rf", "all"], help="Which model to train")
	parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to write outputs")
	parser.add_argument("--sr", type=int, default=22050, help="Audio sample rate for loading")
	parser.add_argument("--duration", type=float, default=30.0, help="Seconds to load per track (None for full)")
	parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
	parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
	parser.add_argument("--n-jobs", type=int, default=-1, help="Parallelism for training where applicable")
	parser.add_argument("--grid", action="store_true", help="Run small GridSearchCV for selected model(s)")
	parser.add_argument("--save-features", action="store_true", help="Save extracted features to CSV in output dir")
	parser.add_argument("--max-files-per-class", type=int, default=None, help="Cap files per singer for quicker experiments")

	args = parser.parse_args()

	start_time = time.time()

	os.makedirs(args.output_dir, exist_ok=True)

	feature_config = FeatureConfig(sample_rate=args.sr, duration_s=(None if args.duration <= 0 else args.duration))

	print(f"Scanning dataset in: {args.data_dir}")
	features_df = build_features_dataframe(
		args.data_dir,
		config=feature_config,
		max_files_per_class=args.max_files_per_class,
	)
	print(f"Extracted features for {len(features_df)} tracks with {features_df['label'].nunique()} singers")

	if args.save_features:
		features_csv_path = os.path.join(args.output_dir, "features.csv")
		features_df.to_csv(features_csv_path, index=False)
		print(f"Saved features to {features_csv_path}")

	feature_columns = [c for c in features_df.columns if c not in {"label", "path"}]
	X_all = features_df[feature_columns].to_numpy(dtype=np.float32)
	labels = features_df["label"].astype(str).to_numpy()

	label_encoder = LabelEncoder()
	y_all = label_encoder.fit_transform(labels)

	X_train, X_test, y_train, y_test = train_test_split(
		X_all,
		y_all,
		test_size=args.test_size,
		stratify=y_all,
		random_state=args.random_state,
	)

	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	models_to_run = [args.model] if args.model != "all" else ["knn", "svm", "rf"]
	results: Dict[str, Dict[str, object]] = {}
	trained_models: Dict[str, object] = {}

	for model_key in models_to_run:
		print(f"Training model: {model_key} ({'with' if args.grid else 'no'} grid search)")
		m_key, fitted_model, metrics_dict = train_and_evaluate(
			X_train_scaled,
			y_train,
			X_test_scaled,
			y_test,
			model_key=model_key,
			run_grid_search=args.grid,
			random_state=args.random_state,
			n_jobs=args.n_jobs,
		)
		results[m_key] = metrics_dict
		trained_models[m_key] = fitted_model
		print(f"{m_key}: test accuracy={metrics_dict['test_accuracy']:.4f} | f1_macro={metrics_dict['test_f1_macro']:.4f}")

	best_model_key = max(results, key=lambda k: float(results[k]["test_accuracy"]))
	best_model = trained_models[best_model_key]
	print(f"Best model: {best_model_key}")

	model_out = os.path.join(args.output_dir, "model.joblib")
	scaler_out = os.path.join(args.output_dir, "scaler.joblib")
	le_out = os.path.join(args.output_dir, "label_encoder.joblib")
	metrics_out = os.path.join(args.output_dir, "metrics.json")

	joblib.dump(best_model, model_out)
	joblib.dump(scaler, scaler_out)
	joblib.dump(label_encoder, le_out)

	columns_out = os.path.join(args.output_dir, "feature_columns.json")
	with open(columns_out, "w", encoding="utf-8") as fcols:
		json.dump(feature_columns, fcols)

	payload = {
		"best_model": best_model_key,
		"results": results,
		"num_train": int(X_train.shape[0]),
		"num_test": int(X_test.shape[0]),
		"labels": list(label_encoder.classes_.astype(str)),
	}
	with open(metrics_out, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)

	elapsed = time.time() - start_time
	print(f"Saved model to {model_out}")
	print(f"Saved scaler to {scaler_out}")
	print(f"Saved label encoder to {le_out}")
	print(f"Saved metrics to {metrics_out}")
	print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
	main()