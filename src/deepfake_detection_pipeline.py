"""Reproducible training and evaluation pipeline exported from the thesis notebook.

Run cells selectively in an IDE or execute the cleaned notebook for the intended workflow.
"""

# %% [markdown]
# # Thesis-Compliant CNN-LSTM Deepfake Video Detector
#
# This notebook is cleaned for the thesis guidelines. It uses video-first dataset splitting, train-only augmentation, full evaluation metrics, lightweight pretrained baselines, computational efficiency reporting, XAI, and a prototype inference function.

# %% [markdown]
# > Repository note: this public copy removes execution outputs, replaces machine-specific paths with environment variables, and guards destructive regeneration behind `REBUILD_DERIVED_DATA=1`. The original research notebook remains unchanged.
#
# Set `FACEFORENSICS_ROOT` to the FaceForensics++ C23 directory and optionally set `DEEPFAKE_PROJECT_ROOT` for generated artifacts.

# %%
import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))

# %%
# Step 0: Setup, reproducibility, and folders

import os
import json
import random
import shutil
import math
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, brier_score_loss,
    roc_curve, auc
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# GPU speed settings for Kaggle T4 x2.
# OpenCV extraction still uses CPU, but TensorFlow training/prediction should use GPU.
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Could not set memory growth:", e)

# Use float32 for stable thesis runs. Mixed precision caused NaN/collapse in later experiments.
tf.keras.mixed_precision.set_global_policy("float32")
print("Precision policy:", tf.keras.mixed_precision.global_policy())

try:
    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()
except Exception as e:
    print("Could not create MirroredStrategy, using default strategy:", e)
    strategy = tf.distribute.get_strategy()

NUM_REPLICAS = strategy.num_replicas_in_sync
print("Distribution strategy:", type(strategy).__name__)
print("Replicas in sync:", NUM_REPLICAS)


PROJECT_ROOT = os.environ.get("DEEPFAKE_PROJECT_ROOT", str(Path.cwd() / "run"))

ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODEL_DIR = os.path.join(ARTIFACT_DIR, "models")
PLOT_DIR = os.path.join(ARTIFACT_DIR, "plots")
RESULT_DIR = os.path.join(ARTIFACT_DIR, "results")
FRAME_DIR = os.path.join(PROJECT_ROOT, "frame_dataset")
CLIP_DIR = os.path.join(PROJECT_ROOT, "clips")
FEATURE_DIR = os.path.join(ARTIFACT_DIR, "features")

for folder in [ARTIFACT_DIR, MODEL_DIR, PLOT_DIR, RESULT_DIR, FRAME_DIR, CLIP_DIR, FEATURE_DIR]:
    os.makedirs(folder, exist_ok=True)

print("TensorFlow:", tf.__version__)
print("GPUs:", gpus)
print("Artifact folder:", ARTIFACT_DIR)

# %%


# Step 1: Dataset path - LOCAL WINDOWS VERSION

import os

dataset_path = os.environ.get("FACEFORENSICS_ROOT", str(Path.cwd() / "data" / "FaceForensics++_C23"))

real_video_dir = os.path.join(dataset_path, "original")
fake_video_dir = os.path.join(dataset_path, "Deepfakes")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

if not os.path.exists(real_video_dir):
    raise FileNotFoundError(f"Real folder not found: {real_video_dir}")

if not os.path.exists(fake_video_dir):
    raise FileNotFoundError(f"Fake folder not found: {fake_video_dir}")

real_videos_all = sorted([f for f in os.listdir(real_video_dir) if f.endswith(".mp4")])
fake_videos_all = sorted([f for f in os.listdir(fake_video_dir) if f.endswith(".mp4")])

print("Dataset path:", dataset_path)
print("Real path:", real_video_dir)
print("Fake path:", fake_video_dir)
print("Real videos:", len(real_videos_all))
print("Fake videos:", len(fake_videos_all))

# %% [markdown]
# ## 1. Dataset Split
#
# Guideline rule followed here: split the dataset first into **70% train, 20% validation, 10% test**. Frame and clip extraction happen after this split to avoid data leakage.

# %%
# Step 2: Video-first 70/20/10 split

# Final thesis setting: use the full available FaceForensics++ C23 split.
# If Kaggle time runs out during debugging, temporarily set USE_FULL_DATASET = False.
USE_FULL_DATASET = True
DEBUG_REAL = 300
DEBUG_FAKE = 300

N_REAL = len(real_videos_all) if USE_FULL_DATASET else DEBUG_REAL
N_FAKE = len(fake_videos_all) if USE_FULL_DATASET else DEBUG_FAKE

random.seed(SEED)
real_videos = real_videos_all.copy()
fake_videos = fake_videos_all.copy()
random.shuffle(real_videos)
random.shuffle(fake_videos)

real_videos = real_videos[:N_REAL]
fake_videos = fake_videos[:N_FAKE]

def split_videos(items, train_ratio=0.70, val_ratio=0.20):
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return items[:n_train], items[n_train:n_train+n_val], items[n_train+n_val:]

real_train, real_val, real_test = split_videos(real_videos)
fake_train, fake_val, fake_test = split_videos(fake_videos)

split_manifest = {
    "use_full_dataset": USE_FULL_DATASET,
    "n_real": N_REAL,
    "n_fake": N_FAKE,
    "split_rule": "70% train, 20% validation, 10% test by video filename before frame/clip extraction",
    "real_train": real_train,
    "real_val": real_val,
    "real_test": real_test,
    "fake_train": fake_train,
    "fake_val": fake_val,
    "fake_test": fake_test,
}

with open(os.path.join(RESULT_DIR, "video_split_manifest.json"), "w") as f:
    json.dump(split_manifest, f, indent=2)

print("USE_FULL_DATASET:", USE_FULL_DATASET)
print("Real split train/val/test:", len(real_train), len(real_val), len(real_test))
print("Fake split train/val/test:", len(fake_train), len(fake_val), len(fake_test))
print("Total train/val/test videos:", len(real_train)+len(fake_train), len(real_val)+len(fake_val), len(real_test)+len(fake_test))

# %%
import os
import shutil

for name, path in {
    "PROJECT_ROOT": PROJECT_ROOT,
    "FRAME_DIR": FRAME_DIR,
    "CLIP_DIR": CLIP_DIR,
    "ARTIFACT_DIR": ARTIFACT_DIR,
    "FEATURE_DIR": FEATURE_DIR,
}.items():
    print(name, "=>", path)

total, used, free = shutil.disk_usage(PROJECT_ROOT)
print("\nFree space:", round(free / (1024**3), 2), "GB")

# %%
# Step 2B: Face crop helper

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def crop_face_or_center(frame_rgb, output_size=(224, 224), margin=0.35):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60)
    )

    h, w = frame_rgb.shape[:2]

    if len(faces) > 0:
        # choose largest face
        x, y, fw, fh = max(faces, key=lambda box: box[2] * box[3])

        mx = int(fw * margin)
        my = int(fh * margin)

        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + fw + mx)
        y2 = min(h, y + fh + my)

        crop = frame_rgb[y1:y2, x1:x2]
    else:
        # fallback center square crop
        side = min(h, w)
        x1 = (w - side) // 2
        y1 = (h - side) // 2
        crop = frame_rgb[y1:y1+side, x1:x1+side]

    crop = cv2.resize(crop, output_size)
    return crop

# %% [markdown]
# ## 2. Frame Dataset For CNN Training
#
# Frames are extracted after the video split. Augmentation is applied only inside the training data pipeline, never before splitting.

# %%
# Step 3: Extract frames from the already-split videos

IMAGE_SIZE = (224, 224)
FRAMES_PER_VIDEO = 32   # max-validation run: more frames per video for stronger generalization

REBUILD_DERIVED_DATA = os.environ.get("REBUILD_DERIVED_DATA", "0") == "1"
if REBUILD_DERIVED_DATA and os.path.exists(FRAME_DIR):
    shutil.rmtree(FRAME_DIR)

for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        os.makedirs(os.path.join(FRAME_DIR, split, label), exist_ok=True)

def extract_evenly_spaced_frames(video_path, out_dir, label, frames_per_video=32, image_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return 0

    frame_indices = np.linspace(0, max(total - 1, 0), frames_per_video, dtype=int)
    video_id = os.path.basename(video_path).replace(".mp4", "")
    saved = 0

    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = crop_face_or_center(frame, output_size=image_size, margin=0.35)
        out_path = os.path.join(out_dir, f"{label}_{video_id}_frame_{idx}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        saved += 1

    cap.release()
    return saved

def build_frame_split(videos, src_dir, split, label):
    out_dir = os.path.join(FRAME_DIR, split, label)
    total_saved = 0
    for video_name in tqdm(videos, desc=f"frames {split}/{label}"):
        total_saved += extract_evenly_spaced_frames(
            os.path.join(src_dir, video_name),
            out_dir,
            label,
            FRAMES_PER_VIDEO,
            IMAGE_SIZE
        )
    print(split, label, "frames:", total_saved)

build_frame_split(real_train, real_video_dir, "train", "real")
build_frame_split(real_val, real_video_dir, "val", "real")
build_frame_split(real_test, real_video_dir, "test", "real")
build_frame_split(fake_train, fake_video_dir, "train", "fake")
build_frame_split(fake_val, fake_video_dir, "val", "fake")
build_frame_split(fake_test, fake_video_dir, "test", "fake")

for split in ["train", "val", "test"]:
    print(split, "real:", len(os.listdir(os.path.join(FRAME_DIR, split, "real"))),
          "fake:", len(os.listdir(os.path.join(FRAME_DIR, split, "fake"))))

# %%
# Step 4: Frame dataloaders with augmentation only on training data

BATCH_SIZE_FRAME = 32 * max(NUM_REPLICAS, 1)
AUTOTUNE = tf.data.AUTOTUNE
print("Frame batch size:", BATCH_SIZE_FRAME)

raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(FRAME_DIR, "train"),
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE_FRAME,
    label_mode="binary"
)
raw_train_eval_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(FRAME_DIR, "train"),
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE_FRAME,
    label_mode="binary",
    shuffle=False
)
raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(FRAME_DIR, "val"),
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE_FRAME,
    label_mode="binary",
    shuffle=False
)
raw_test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(FRAME_DIR, "test"),
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE_FRAME,
    label_mode="binary",
    shuffle=False
)

class_names = raw_train_ds.class_names
print("Class names:", class_names)
print("Keras mapping: fake=0, real=1")

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=SEED),
    tf.keras.layers.RandomRotation(0.05, seed=SEED),
    tf.keras.layers.RandomZoom(0.10, seed=SEED),
    tf.keras.layers.RandomContrast(0.10, seed=SEED),
], name="train_only_augmentation")

normalization = tf.keras.layers.Rescaling(1./255)


def preprocess_train(x, y):
    x = data_augmentation(x, training=True)
    x = normalization(x)
    y = tf.cast(y, tf.float32)
    return x, y


def preprocess_eval(x, y):
    x = normalization(x)
    y = tf.cast(y, tf.float32)
    return x, y




train_ds_nomix = raw_train_ds.map(preprocess_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
train_ds = raw_train_ds.map(preprocess_train,num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = raw_val_ds.map(preprocess_eval, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_ds = raw_test_ds.map(preprocess_eval, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
train_eval_ds = raw_train_eval_ds.map(preprocess_eval, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

print("Training augmentation is applied only to train_ds. MixUp removed.")
print("train_eval_ds is clean and used only for final reporting.")

# %% [markdown]
# ## 3. Proposed Custom CNN
#
# This is the proposed lightweight custom CNN backbone. It is trained from scratch and later used as the CNN part of the CNN-LSTM/CNN-temporal hybrid model.

# %%
# Step 5: Proposed stronger custom CNN backbone
# This replaces the previous tiny backbone. It is still custom and lightweight, but has enough capacity to learn useful frame features.

from tensorflow.keras import layers, models, regularizers

def se_block(x, filters, ratio=8, name="se"):
    se = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    se = layers.Reshape((1, 1, filters), name=f"{name}_reshape")(se)
    se = layers.Dense(max(filters // ratio, 1), activation="relu", name=f"{name}_squeeze")(se)
    se = layers.Dense(filters, activation="sigmoid", name=f"{name}_excite")(se)
    return layers.Multiply(name=f"{name}_scale")([x, se])

def conv_block(x, filters, name, separable=False, pool=True):
    Conv = layers.SeparableConv2D if separable else layers.Conv2D

    if separable:
        x = Conv(
            filters, 3, padding="same", use_bias=False,
            depthwise_regularizer=regularizers.l2(1e-4),
            pointwise_regularizer=regularizers.l2(1e-4),
            name=f"{name}_sepconv1"
        )(x)
    else:
        x = Conv(filters, 3, padding="same", use_bias=False,
                 kernel_regularizer=regularizers.l2(1e-4), name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)

    if separable:
        x = Conv(
            filters, 3, padding="same", use_bias=False,
            depthwise_regularizer=regularizers.l2(1e-4),
            pointwise_regularizer=regularizers.l2(1e-4),
            name=f"{name}_sepconv2"
        )(x)
    else:
        x = Conv(filters, 3, padding="same", use_bias=False,
                 kernel_regularizer=regularizers.l2(1e-4), name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)

    x = se_block(x, filters, name=f"{name}_se")
    if pool:
        x = layers.MaxPooling2D(name=f"{name}_pool")(x)
    return x


def build_custom_strong_cnn(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape, name="frame_input")

    x = conv_block(inputs, 32, "block1", separable=False)
    x = conv_block(x, 64, "block2", separable=False)
    x = conv_block(x, 128, "block3", separable=True)
    x = conv_block(x, 192, "block4", separable=True)
    x = conv_block(x, 256, "block5", separable=True)

    x = layers.GlobalAveragePooling2D(name="cnn_gap")(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="feature_dense")(x)
    x = layers.BatchNormalization(name="feature_bn")(x)
    x = layers.Dropout(0.5, name="feature_dropout")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="compact_feature_dense")(x)
    x = layers.Dropout(0.4, name="classifier_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32", name="real_probability")(x)

    return models.Model(inputs, outputs, name="proposed_custom_maxval_cnn")

with strategy.scope():
    custom_cnn = build_custom_strong_cnn()
    custom_cnn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )
custom_cnn.summary()
print("Custom CNN parameters:", custom_cnn.count_params())

# %%
# Step 6: Train proposed custom CNN

custom_cnn_path = os.path.join(MODEL_DIR, "proposed_custom_oldstyle_aug_reg_cnn.keras")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(custom_cnn_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.TerminateOnNaN(),
]

history_custom_cnn = custom_cnn.fit(
    train_ds,
    validation_data=val_ds,
    epochs=40,
    callbacks=callbacks,
    verbose=1
)

with open(os.path.join(RESULT_DIR, "custom_cnn_history.json"), "w") as f:
    json.dump(history_custom_cnn.history, f)

custom_cnn = tf.keras.models.load_model(custom_cnn_path)
print("Loaded best custom CNN from:", custom_cnn_path)

# %%
# Step 7: Evaluation helper for binary classifiers
# Important: because the thesis goal is video detection, the Custom CNN is evaluated two ways:
# 1) frame-level accuracy
# 2) video-level accuracy by averaging all frame predictions from the same original video


def specificity_from_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp + 1e-8)


def collect_image_scores(model, dataset):
    y_true, y_score = [], []
    for x, y in dataset:
        scores = model.predict(x, verbose=0).reshape(-1)
        scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
        y_score.extend(scores)
        y_true.extend(y.numpy().reshape(-1))
    return np.asarray(y_true).astype(int), np.asarray(y_score).astype(np.float32)


def frame_path_to_video_id(path):
    name = os.path.basename(path).replace(".jpg", "")
    label = os.path.basename(os.path.dirname(path))

    prefix = label + "_"
    if name.startswith(prefix):
        name = name[len(prefix):]

    if "_frame_" in name:
        name = name.split("_frame_")[0]

    # IMPORTANT: include label to prevent real/fake videos with same filename merging
    return f"{label}__{name}"


def summarize_scores(y_true, y_score, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(np.float32)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity_from_confusion(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_score),
        "brier_score": brier_score_loss(y_true, y_score),
    }


def make_frame_and_video_predictions(model, dataset, file_paths, split_name):
    y_true, y_score = collect_image_scores(model, dataset)
    frame_df = pd.DataFrame({
        "frame_path": file_paths,
        "video_id": [frame_path_to_video_id(p) for p in file_paths],
        "true_label": y_true,
        "score_real": y_score,
    })
    video_df = frame_df.groupby("video_id").agg(
        true_label=("true_label", "first"),
        mean_score_real=("score_real", "mean"),
        median_score_real=("score_real", "median"),
        num_frames=("score_real", "count")
    ).reset_index()
    frame_df.to_csv(os.path.join(RESULT_DIR, f"custom_cnn_{split_name}_frame_predictions.csv"), index=False)
    video_df.to_csv(os.path.join(RESULT_DIR, f"custom_cnn_{split_name}_video_predictions_raw.csv"), index=False)
    return frame_df, video_df


def choose_threshold_from_video_validation(video_df):
    rows = []
    y_true = video_df["true_label"].astype(int).values
    y_score = video_df["mean_score_real"].values
    for threshold in np.linspace(0.10, 0.90, 81):
        rows.append(summarize_scores(y_true, y_score, threshold))
    threshold_df = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "f1", "accuracy"], ascending=False
    ).reset_index(drop=True)
    threshold_df.to_csv(os.path.join(RESULT_DIR, "custom_cnn_video_validation_threshold_search.csv"), index=False)
    best_threshold = float(threshold_df.loc[0, "threshold"])
    print("Best Custom CNN video threshold selected on validation:", best_threshold)
    print(threshold_df.head(10))
    return best_threshold


def evaluate_frame_scores(frame_df, model_name, split_name, threshold=0.5):
    y_true = frame_df["true_label"].astype(int).values
    y_score = frame_df["score_real"].values
    y_pred = (y_score >= threshold).astype(int)
    metrics = {"model": model_name, "level": f"frame_{split_name}", **summarize_scores(y_true, y_score, threshold), "params": int(custom_cnn.count_params())}
    print("\n" + "="*80)
    print(f"{model_name} - frame {split_name}")
    print("="*80)
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("AUC:", metrics["auc"])
    return metrics


def evaluate_video_scores(video_df, model_name, split_name, threshold):
    y_true = video_df["true_label"].astype(int).values
    y_score = video_df["mean_score_real"].values
    y_pred = (y_score >= threshold).astype(int)
    metrics = {"model": model_name, "level": f"video_{split_name}", **summarize_scores(y_true, y_score, threshold), "params": int(custom_cnn.count_params())}
    out_df = video_df.copy()
    out_df["pred_label"] = y_pred
    out_df.to_csv(os.path.join(RESULT_DIR, f"custom_cnn_{split_name}_video_predictions.csv"), index=False)
    print("\n" + "="*80)
    print(f"{model_name} - VIDEO {split_name}")
    print("="*80)
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("Video confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Video AUC:", metrics["auc"])
    print("Video threshold:", threshold)
    return metrics


train_frame_df, train_video_df = make_frame_and_video_predictions(custom_cnn, train_eval_ds, raw_train_eval_ds.file_paths, "train")
val_frame_df, val_video_df = make_frame_and_video_predictions(custom_cnn, val_ds, raw_val_ds.file_paths, "val")
test_frame_df, test_video_df = make_frame_and_video_predictions(custom_cnn, test_ds, raw_test_ds.file_paths, "test")

custom_cnn_video_threshold = choose_threshold_from_video_validation(val_video_df)

custom_cnn_train_frame_metrics = evaluate_frame_scores(train_frame_df, "Proposed Custom CNN", "train", threshold=0.5)
custom_cnn_val_frame_metrics = evaluate_frame_scores(val_frame_df, "Proposed Custom CNN", "val", threshold=0.5)
custom_cnn_metrics = evaluate_frame_scores(test_frame_df, "Proposed Custom CNN", "test", threshold=0.5)

custom_cnn_train_video_metrics = evaluate_video_scores(train_video_df, "Proposed Custom CNN", "train", threshold=custom_cnn_video_threshold)
custom_cnn_val_video_metrics = evaluate_video_scores(val_video_df, "Proposed Custom CNN", "val", threshold=custom_cnn_video_threshold)
custom_cnn_test_video_metrics = evaluate_video_scores(test_video_df, "Proposed Custom CNN", "test", threshold=custom_cnn_video_threshold)

print("\nCustom CNN frame-level accuracies:")
print("Train:", custom_cnn_train_frame_metrics["accuracy"])
print("Validation:", custom_cnn_val_frame_metrics["accuracy"])
print("Test:", custom_cnn_metrics["accuracy"])
print("Frame train-validation gap:", custom_cnn_train_frame_metrics["accuracy"] - custom_cnn_val_frame_metrics["accuracy"])

print("\nCustom CNN VIDEO-level accuracies - use these for the thesis video-detection discussion:")
print("Train:", custom_cnn_train_video_metrics["accuracy"])
print("Validation:", custom_cnn_val_video_metrics["accuracy"])
print("Test:", custom_cnn_test_video_metrics["accuracy"])
print("Video train-validation gap:", custom_cnn_train_video_metrics["accuracy"] - custom_cnn_val_video_metrics["accuracy"])

pd.DataFrame([
    {**custom_cnn_train_frame_metrics, "split": "train"},
    {**custom_cnn_val_frame_metrics, "split": "validation"},
    {**custom_cnn_metrics, "split": "test"},
    {**custom_cnn_train_video_metrics, "split": "train"},
    {**custom_cnn_val_video_metrics, "split": "validation"},
    {**custom_cnn_test_video_metrics, "split": "test"},
]).to_csv(os.path.join(RESULT_DIR, "custom_cnn_frame_and_video_accuracy_report.csv"), index=False)

fpr, tpr, _ = roc_curve(test_frame_df["true_label"].astype(int).values, test_frame_df["score_real"].values)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"AUC = {custom_cnn_metrics['auc']:.4f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Custom CNN Frame Test")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, "custom_cnn_test_roc_curve.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Lightweight Pretrained Baseline Comparison
#
# Guideline rule followed here: the proposed lightweight model is compared with well-known lightweight pretrained models. This notebook includes EfficientNetB0 and MobileNetV2, which are available directly in Keras.

# %%
RUN_PRETRAINED_BASELINES = False
baseline_results = []
baseline_models = {}

# # Step 8: Lightweight pretrained baselines
# # EfficientNetB0 and MobileNetV2 are included because they are standard lightweight Keras baselines.

# RUN_PRETRAINED_BASELINES = True
# BASELINE_EPOCHS = 8

# from tensorflow.keras import applications

# baseline_results = []
# baseline_models = {}

# def build_pretrained_baseline(name, input_shape=(224, 224, 3)):
#     if name == "EfficientNetB0":
#         base = applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
#         preprocess = applications.efficientnet.preprocess_input
#     elif name == "MobileNetV2":
#         base = applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
#         preprocess = applications.mobilenet_v2.preprocess_input
#     else:
#         raise ValueError(name)

#     base.trainable = False
#     inputs = layers.Input(shape=input_shape)
#     x = layers.Lambda(preprocess, name=f"{name}_preprocess")(inputs * 255.0)
#     x = base(x, training=False)
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dropout(0.4)(x)
#     outputs = layers.Dense(1, activation="sigmoid")(x)
#     model = models.Model(inputs, outputs, name=f"baseline_{name.lower()}")
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#         loss="binary_crossentropy",
#         metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
#     )
#     return model

# if RUN_PRETRAINED_BASELINES:
#     for baseline_name in ["EfficientNetB0", "MobileNetV2"]:
#         print("\nTraining baseline:", baseline_name)
#         baseline_model = build_pretrained_baseline(baseline_name)
#         baseline_path = os.path.join(MODEL_DIR, f"baseline_{baseline_name}.keras")
#         baseline_callbacks = [
#             tf.keras.callbacks.ModelCheckpoint(baseline_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
#             tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True, verbose=1),
#         ]
#         baseline_model.fit(train_ds, validation_data=val_ds, epochs=BASELINE_EPOCHS, callbacks=baseline_callbacks, verbose=1)
#         # EarlyStopping restores the best weights, so we can evaluate this in-memory model.
#         # This avoids Lambda deserialization issues in newer Keras versions.
#         metrics = evaluate_binary_image_model(baseline_model, test_ds, baseline_name, f"baseline_{baseline_name}")
#         baseline_results.append(metrics)
#         baseline_models[baseline_name] = baseline_model
# else:
#     print("Pretrained baseline training skipped. Set RUN_PRETRAINED_BASELINES=True for thesis comparison runs.")

# %% [markdown]
# ## 5. Computational Efficiency
#
# Guideline rule followed here: parameters, FLOPs, and MACs are reported for the proposed model and baselines.

# %%
# Step 9: Computational efficiency - parameters, FLOPs, and MACs

# FLOPs are estimated with TensorFlow graph profiling. MACs are approximated as FLOPs / 2.
def estimate_flops(model, input_shape=(1, 224, 224, 3)):
    try:
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        concrete = tf.function(model).get_concrete_function(tf.TensorSpec(input_shape, tf.float32))
        frozen_func = convert_variables_to_constants_v2(concrete)
        graph_def = frozen_func.graph.as_graph_def()

        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name="")
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
            return int(flops.total_float_ops) if flops is not None else np.nan
    except Exception as e:
        print("FLOPs estimation failed for", model.name, "->", e)
        return np.nan

efficiency_rows = []
models_for_efficiency = [("Proposed Custom CNN", custom_cnn)]

if RUN_PRETRAINED_BASELINES and baseline_results:
    for baseline_name, baseline_model in baseline_models.items():
        models_for_efficiency.append((baseline_name, baseline_model))

for model_name, model in models_for_efficiency:
    flops = estimate_flops(model)
    efficiency_rows.append({
        "model": model_name,
        "parameters": int(model.count_params()),
        "FLOPs": flops,
        "MACs_approx": flops / 2 if not np.isnan(flops) else np.nan
    })

efficiency_df = pd.DataFrame(efficiency_rows)
print(efficiency_df)
efficiency_df.to_csv(os.path.join(RESULT_DIR, "computational_efficiency.csv"), index=False)

# %% [markdown]
# ## 6. Video-Level Hybrid Model
#
# The trained custom CNN becomes a feature extractor. A temporal model then learns from sequences of CNN features extracted from video clips.

# %%
# Step 10: Extract video clips from the already-split videos

IMG_SIZE_CLIP = 160
T = 24
CLIPS_PER_VIDEO = 5

if REBUILD_DERIVED_DATA and os.path.exists(CLIP_DIR):
    shutil.rmtree(CLIP_DIR)

for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        os.makedirs(os.path.join(CLIP_DIR, split, label), exist_ok=True)

def read_multiple_video_clips(video_path, T=24, img_size=160, clips_per_video=5):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    start_points = np.linspace(0, max(total - T, 0), clips_per_video, dtype=int) if total > T else [0]
    clips = []

    for start in start_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        frames = []
        for _ in range(T):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = crop_face_or_center(frame, output_size=(img_size, img_size), margin=0.35)
            frames.append(frame)
        if not frames:
            continue
        while len(frames) < T:
            frames.append(frames[-1])
        clips.append(np.stack(frames, axis=0).astype(np.uint8))

    cap.release()
    return clips

def build_clip_split(videos, src_dir, split, label):
    out_dir = os.path.join(CLIP_DIR, split, label)
    saved = 0
    for video_name in tqdm(videos, desc=f"clips {split}/{label}"):
        video_id = video_name.replace(".mp4", "")
        clips = read_multiple_video_clips(os.path.join(src_dir, video_name), T, IMG_SIZE_CLIP, CLIPS_PER_VIDEO)
        for clip_idx, clip in enumerate(clips):
            np.save(os.path.join(out_dir, f"{video_id}_clip_{clip_idx}.npy"), clip)
            saved += 1
    print(split, label, "clips:", saved)

build_clip_split(real_train, real_video_dir, "train", "real")
build_clip_split(real_val, real_video_dir, "val", "real")
build_clip_split(real_test, real_video_dir, "test", "real")
build_clip_split(fake_train, fake_video_dir, "train", "fake")
build_clip_split(fake_val, fake_video_dir, "val", "fake")
build_clip_split(fake_test, fake_video_dir, "test", "fake")

for split in ["train", "val", "test"]:
    print(split, "real:", len(os.listdir(os.path.join(CLIP_DIR, split, "real"))),
          "fake:", len(os.listdir(os.path.join(CLIP_DIR, split, "fake"))))

# %%
# Step 11: Precompute custom CNN feature sequences for clips
# Important: feature files must be recomputed whenever the CNN backbone change

with strategy.scope():
    feature_extractor = tf.keras.Model(
        inputs=custom_cnn.input,
        outputs=custom_cnn.get_layer("feature_dense").output,
        name="custom_cnn_feature_extractor"
    )
feature_extractor.trainable = False

FEATURE_BATCH_SIZE = 128 * max(NUM_REPLICAS, 1)      # frame batch inside CNN prediction
CLIP_FEATURE_BATCH_SIZE = 16                         # number of clips loaded per feature pass
FORCE_RECOMPUTE_FEATURES = False
FEATURE_TAG = f"customcnn_params{custom_cnn.count_params()}_T{T}_C{CLIPS_PER_VIDEO}"

print("Feature frame batch size:", FEATURE_BATCH_SIZE)
print("Clip feature batch size:", CLIP_FEATURE_BATCH_SIZE)

def list_npy_files(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npy")])

def collect_clip_split(split):
    real_files = list_npy_files(os.path.join(CLIP_DIR, split, "real"))
    fake_files = list_npy_files(os.path.join(CLIP_DIR, split, "fake"))
    files = real_files + fake_files
    labels = np.asarray([1] * len(real_files) + [0] * len(fake_files), dtype=np.float32)
    return files, labels

train_files, train_labels = collect_clip_split("train")
val_files, val_labels = collect_clip_split("val")
test_files, test_labels = collect_clip_split("test")

def load_and_resize_clip(path):
    clip = np.load(path).astype(np.float32) / 255.0
    clip = tf.image.resize(clip, IMAGE_SIZE).numpy()
    return clip

def precompute_features(files, labels, split):
    features_path = os.path.join(FEATURE_DIR, f"{split}_features_{FEATURE_TAG}.npy")
    labels_path = os.path.join(FEATURE_DIR, f"{split}_labels_{FEATURE_TAG}.npy")
    files_path = os.path.join(FEATURE_DIR, f"{split}_files_{FEATURE_TAG}.json")

    if (not FORCE_RECOMPUTE_FEATURES) and os.path.exists(features_path) and os.path.exists(labels_path) and os.path.exists(files_path):
        print("Loading cached features:", split, FEATURE_TAG)
        with open(files_path) as f:
            cached_files = json.load(f)
        return np.load(features_path), np.load(labels_path), cached_files

    print("Recomputing features with batched GPU prediction:", split, FEATURE_TAG)
    all_features = []

    for start in tqdm(range(0, len(files), CLIP_FEATURE_BATCH_SIZE), desc=f"features {split}"):
        batch_files = files[start:start + CLIP_FEATURE_BATCH_SIZE]
        clip_batch = np.stack([load_and_resize_clip(path) for path in batch_files], axis=0).astype(np.float32)
        n_clips = clip_batch.shape[0]

        # Convert (clips, frames, H, W, C) -> (clips*frames, H, W, C) for one large CNN call.
        flat_frames = clip_batch.reshape((-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        flat_features = feature_extractor.predict(flat_frames, batch_size=FEATURE_BATCH_SIZE, verbose=0)
        clip_features = flat_features.reshape((n_clips, T, -1)).astype(np.float32)
        all_features.append(clip_features)

    features = np.concatenate(all_features, axis=0)
    np.save(features_path, features)
    np.save(labels_path, labels)
    with open(files_path, "w") as f:
        json.dump(files, f)

    print(split, features.shape)
    return features, labels, files

train_features, train_labels_np, train_files = precompute_features(train_files, train_labels, "train")
val_features, val_labels_np, val_files = precompute_features(val_files, val_labels, "val")
test_features, test_labels_np, test_files = precompute_features(test_files, test_labels, "test")

print("Feature extractor output dimension:", feature_extractor.output_shape)
print("Feature shapes:", train_features.shape, val_features.shape, test_features.shape)

# %%
# Step 12: Train CNN + attention-enhanced BiLSTM temporal model

BATCH_SIZE_FEATURE = 16 * max(NUM_REPLICAS, 1)
AUTOTUNE = tf.data.AUTOTUNE
print("Temporal feature batch size:", BATCH_SIZE_FEATURE)

def make_feature_dataset(features, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE_FEATURE).prefetch(AUTOTUNE)

train_feature_ds = make_feature_dataset(train_features, train_labels_np, shuffle=True)
val_feature_ds = make_feature_dataset(val_features, val_labels_np, shuffle=False)
test_feature_ds = make_feature_dataset(test_features, test_labels_np, shuffle=False)

feature_dim = train_features.shape[-1]

def build_bilstm_temporal_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape, name="cnn_feature_sequence")
    x = tf.keras.layers.LayerNormalization(name="feature_norm")(inputs)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True),
        name="bilstm_1"
    )(x)
    x = tf.keras.layers.Dropout(0.50, name="dropout_1")(x)

    avg_pool = tf.keras.layers.GlobalAveragePooling1D(name="avg_pool")(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D(name="max_pool")(x)
    x = tf.keras.layers.Concatenate(name="temporal_pooling")([avg_pool, max_pool])

    x = tf.keras.layers.Dense(64, activation="relu", name="temporal_dense_1")(x)
    x = tf.keras.layers.Dropout(0.50, name="dropout_2")(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        dtype="float32",
        name="real_probability"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name="simplified_cnn_bilstm_video_detector"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model

with strategy.scope():
    temporal_model = build_bilstm_temporal_model((T, feature_dim))
temporal_model.summary()

temporal_model_path = os.path.join(MODEL_DIR, "simplified_cnn_bilstm_video_detector.keras")
temporal_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(temporal_model_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
]

history_temporal = temporal_model.fit(
    train_feature_ds,
    validation_data=val_feature_ds,
    epochs=50,
    callbacks=temporal_callbacks,
    verbose=1
)

with open(os.path.join(RESULT_DIR, "temporal_attention_bilstm_history.json"), "w") as f:
    json.dump(history_temporal.history, f)

temporal_model = tf.keras.models.load_model(temporal_model_path)
print("Loaded best temporal model:", temporal_model_path)

# %%
# Step 13: Validation threshold tuning, clip-level evaluation, video-level evaluation, and confidence intervals

def file_to_video_id(path):
    return os.path.basename(path).split("_clip_")[0]


def summarize_predictions(y_true, y_score, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity_from_confusion(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_score),
        "brier_score": brier_score_loss(y_true, y_score),
    }


def make_video_dataframe(scores, labels, files, threshold):
    clip_df = pd.DataFrame({
        "clip_path": files,
        "video_id": [file_to_video_id(p) for p in files],
        "true_label": labels.astype(int),
        "score_real": scores,
    })
    clip_df["pred_label"] = (clip_df["score_real"] >= threshold).astype(int)

    video_df = clip_df.groupby("video_id").agg(
        true_label=("true_label", "first"),
        mean_score_real=("score_real", "mean"),
        median_score_real=("score_real", "median"),
        min_score_real=("score_real", "min"),
        max_score_real=("score_real", "max"),
        num_clips=("score_real", "count")
    ).reset_index()
    video_df["pred_label"] = (video_df["mean_score_real"] >= threshold).astype(int)
    return clip_df, video_df


def choose_threshold_on_validation(model, val_features, val_labels, val_files):
    val_scores = model.predict(val_features, batch_size=BATCH_SIZE_FEATURE, verbose=0).reshape(-1)
    _, val_video_df = make_video_dataframe(val_scores, val_labels, val_files, threshold=0.5)

    thresholds = np.linspace(0.10, 0.90, 81)
    rows = []
    for threshold in thresholds:
        metrics = summarize_predictions(val_video_df["true_label"], val_video_df["mean_score_real"], threshold)
        rows.append(metrics)

    threshold_df = pd.DataFrame(rows)
    threshold_df = threshold_df.sort_values(["f1", "balanced_accuracy", "auc"], ascending=False).reset_index(drop=True)
    best_threshold = float(threshold_df.loc[0, "threshold"])
    threshold_df.to_csv(os.path.join(RESULT_DIR, "validation_threshold_search.csv"), index=False)

    print("Best threshold selected on validation videos:", best_threshold)
    print(threshold_df.head(10))
    return best_threshold


def bootstrap_metric_ci(y_true, y_score, threshold, metric_fn, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    values = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample_true = y_true[idx]
        sample_score = y_score[idx]
        if len(np.unique(sample_true)) < 2:
            continue
        sample_pred = (sample_score >= threshold).astype(int)
        values.append(metric_fn(sample_true, sample_pred, sample_score))

    if not values:
        return np.nan, np.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def evaluate_temporal_model(model, features, labels, files, threshold):
    scores = model.predict(features, batch_size=BATCH_SIZE_FEATURE, verbose=0).reshape(-1)
    true = labels.astype(int)
    pred = (scores >= threshold).astype(int)

    clip_base = summarize_predictions(true, scores, threshold)
    clip_metrics = {
        "model": "Proposed CNN-Attention-BiLSTM",
        "level": "clip",
        **clip_base,
        "params": int(model.count_params())
    }

    print("Clip-level classification report:")
    print(classification_report(true, pred, digits=4, zero_division=0))
    print("Clip confusion matrix:")
    print(confusion_matrix(true, pred))

    clip_df, video_df = make_video_dataframe(scores, labels, files, threshold)
    video_true = video_df["true_label"].astype(int).values
    video_score = video_df["mean_score_real"].values
    video_pred = (video_score >= threshold).astype(int)

    video_base = summarize_predictions(video_true, video_score, threshold)
    acc_low, acc_high = bootstrap_metric_ci(
        video_true,
        video_score,
        threshold,
        lambda yt, yp, ys: accuracy_score(yt, yp)
    )
    auc_low, auc_high = bootstrap_metric_ci(
        video_true,
        video_score,
        threshold,
        lambda yt, yp, ys: roc_auc_score(yt, ys)
    )

    video_metrics = {
        "model": "Proposed CNN-Attention-BiLSTM",
        "level": "video",
        **video_base,
        "accuracy_ci95_low": acc_low,
        "accuracy_ci95_high": acc_high,
        "auc_ci95_low": auc_low,
        "auc_ci95_high": auc_high,
        "params": int(model.count_params())
    }

    print("\nVideo-level classification report:")
    print(classification_report(video_true, video_pred, digits=4, zero_division=0))
    print("Video confusion matrix:")
    print(confusion_matrix(video_true, video_pred))
    print("Video metrics:", video_metrics)

    clip_df.to_csv(os.path.join(RESULT_DIR, "temporal_clip_predictions.csv"), index=False)
    video_df.to_csv(os.path.join(RESULT_DIR, "temporal_video_predictions.csv"), index=False)

    fpr, tpr, _ = roc_curve(video_true, video_score)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {video_metrics['auc']:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.title("ROC Curve - Video-Level CNN-Attention-BiLSTM")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "temporal_video_roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.show()

    return clip_metrics, video_metrics, clip_df, video_df

best_video_threshold = choose_threshold_on_validation(temporal_model, val_features, val_labels_np, val_files)
temporal_clip_metrics, temporal_video_metrics, clip_predictions_df, video_predictions_df = evaluate_temporal_model(
    temporal_model, test_features, test_labels_np, test_files, threshold=best_video_threshold
)

# %%
## Step 14: EfficientNetB0 fine-tuning experiment


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

base = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)
)

# unfreeze top part
for layer in base.layers[:-50]:
    layer.trainable = False

inputs = layers.Input(shape=(224,224,3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs * 255.0)
x = base(x, training=True)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

efficientnet_ft = models.Model(inputs, outputs)

efficientnet_ft.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

efficientnet_ft.summary()

# %%
## Step 14B: Train EfficientNetB0 fine-tuned model

efficientnet_ft_path = os.path.join(MODEL_DIR, "efficientnetb0_finetuned.keras")

efficientnet_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        efficientnet_ft_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
]

history_efficientnet_ft = efficientnet_ft.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=efficientnet_callbacks,
    verbose=1
)

efficientnet_ft = tf.keras.models.load_model(efficientnet_ft_path)
print("Loaded best EfficientNetB0 fine-tuned model:", efficientnet_ft_path)

# %% [markdown]
# Experiment: EfficientNetB0 fine-tuning
#
# Result:
# Validation AUC remained below 0.50 despite increasing training accuracy.
#
# Conclusion:
# The proposed Custom CNN generalized substantially better on the FaceForensics++ dataset and was therefore selected as the spatial backbone for the temporal model.

# %%
# Step 13B: Final video-level fusion of CNN frame evidence + CNN-BiLSTM temporal evidence
# This is fast: it does not retrain the CNN or LSTM. It combines two useful signals:
# 1) average custom-CNN frame score per video
# 2) average CNN-BiLSTM clip score per video
# The fusion weight and threshold are selected only on validation videos, then tested once on test videos.


def frame_path_to_video_id(path):
    name = os.path.basename(path).replace(".jpg", "")
    parent_label = os.path.basename(os.path.dirname(path))

    prefix = parent_label + "_"
    if name.startswith(prefix):
        name = name[len(prefix):]

    if "_frame_" in name:
        name = name.split("_frame_")[0]
    return name


def predict_frame_dataset_to_video_df(model, dataset, file_paths, batch_size=64):
    scores = model.predict(dataset, verbose=0).reshape(-1)
    rows = []
    for path, score in zip(file_paths, scores):
        label_name = os.path.basename(os.path.dirname(path))
        true_label = 1 if label_name == "real" else 0
        rows.append({
            "video_id": frame_path_to_video_id(path),
            "true_label": true_label,
            "frame_score_real": float(score),
            "frame_path": path,
        })

    frame_df = pd.DataFrame(rows)
    video_df = frame_df.groupby("video_id").agg(
        true_label=("true_label", "first"),
        cnn_mean_score_real=("frame_score_real", "mean"),
        cnn_median_score_real=("frame_score_real", "median"),
        num_frames=("frame_score_real", "count")
    ).reset_index()
    return frame_df, video_df


def temporal_video_scores(model, features, labels, files):
    scores = model.predict(features, batch_size=BATCH_SIZE_FEATURE, verbose=0).reshape(-1)
    _, video_df = make_video_dataframe(scores, labels, files, threshold=0.5)
    return video_df.rename(columns={"mean_score_real": "temporal_mean_score_real"})[
        ["video_id", "true_label", "temporal_mean_score_real", "num_clips"]
    ]


def tune_fusion_on_validation(val_temporal_df, val_frame_video_df):
    merged = val_temporal_df.merge(
        val_frame_video_df[["video_id", "true_label", "cnn_mean_score_real", "num_frames"]],
        on=["video_id", "true_label"],
        how="inner"
    )

    rows = []
    for temporal_weight in np.linspace(0.0, 1.0, 21):
        fused_score = (
            temporal_weight * merged["temporal_mean_score_real"].values
            + (1.0 - temporal_weight) * merged["cnn_mean_score_real"].values
        )
        for threshold in np.linspace(0.10, 0.90, 81):
            metrics = summarize_predictions(merged["true_label"].values, fused_score, threshold)
            rows.append({"temporal_weight": temporal_weight, **metrics})

    search_df = pd.DataFrame(rows).sort_values(
        ["f1", "balanced_accuracy", "auc"], ascending=False
    ).reset_index(drop=True)
    search_df.to_csv(os.path.join(RESULT_DIR, "validation_fusion_search.csv"), index=False)

    best = search_df.iloc[0]
    print("Best fusion selected on validation videos:")
    print(best)
    return float(best["temporal_weight"]), float(best["threshold"]), search_df


def evaluate_video_fusion(test_temporal_df, test_frame_video_df, temporal_weight, threshold):
    merged = test_temporal_df.merge(
        test_frame_video_df[["video_id", "true_label", "cnn_mean_score_real", "num_frames"]],
        on=["video_id", "true_label"],
        how="inner"
    )
    merged["fused_score_real"] = (
        temporal_weight * merged["temporal_mean_score_real"]
        + (1.0 - temporal_weight) * merged["cnn_mean_score_real"]
    )
    merged["pred_label"] = (merged["fused_score_real"] >= threshold).astype(int)

    y_true = merged["true_label"].astype(int).values
    y_score = merged["fused_score_real"].values
    y_pred = merged["pred_label"].values

    metrics = {
        "model": "Final Fusion: Custom CNN + CNN-BiLSTM",
        "level": "video",
        **summarize_predictions(y_true, y_score, threshold),
        "temporal_weight": float(temporal_weight),
        "cnn_frame_weight": float(1.0 - temporal_weight),
        "params": int(custom_cnn.count_params() + temporal_model.count_params())
    }

    acc_low, acc_high = bootstrap_metric_ci(
        y_true, y_score, threshold, lambda yt, yp, ys: accuracy_score(yt, yp)
    )
    auc_low, auc_high = bootstrap_metric_ci(
        y_true, y_score, threshold, lambda yt, yp, ys: roc_auc_score(yt, ys)
    )
    metrics.update({
        "accuracy_ci95_low": acc_low,
        "accuracy_ci95_high": acc_high,
        "auc_ci95_low": auc_low,
        "auc_ci95_high": auc_high,
    })

    print("\nFinal fused video-level classification report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("Final fused video confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Final fused video metrics:", metrics)

    merged.to_csv(os.path.join(RESULT_DIR, "final_fused_video_predictions.csv"), index=False)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.title("ROC Curve - Final Fused Video Detector")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "final_fused_video_roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.show()

    return metrics, merged


val_frame_df, val_frame_video_df = predict_frame_dataset_to_video_df(custom_cnn, val_ds, raw_val_ds.file_paths)
test_frame_df, test_frame_video_df = predict_frame_dataset_to_video_df(custom_cnn, test_ds, raw_test_ds.file_paths)

val_temporal_video_df = temporal_video_scores(temporal_model, val_features, val_labels_np, val_files)
test_temporal_video_df = video_predictions_df.rename(columns={"mean_score_real": "temporal_mean_score_real"})[
    ["video_id", "true_label", "temporal_mean_score_real", "num_clips"]
]

fusion_temporal_weight, fusion_threshold, fusion_search_df = tune_fusion_on_validation(
    val_temporal_video_df,
    val_frame_video_df
)

fusion_video_metrics, fusion_video_predictions_df = evaluate_video_fusion(
    test_temporal_video_df,
    test_frame_video_df,
    fusion_temporal_weight,
    fusion_threshold
)

# %%
## Step 13C: Advanced validation-tuned video aggregation

def load_video_scores():
    cnn_val = pd.read_csv(os.path.join(RESULT_DIR, "custom_cnn_val_video_predictions_raw.csv"))
    cnn_test = pd.read_csv(os.path.join(RESULT_DIR, "custom_cnn_test_video_predictions_raw.csv"))

    temp_val = pd.read_csv(os.path.join(RESULT_DIR, "temporal_video_predictions_val.csv")) if os.path.exists(os.path.join(RESULT_DIR, "temporal_video_predictions_val.csv")) else None
    temp_test = pd.read_csv(os.path.join(RESULT_DIR, "temporal_video_predictions.csv"))

    return cnn_val, cnn_test, temp_val, temp_test

cnn_val, cnn_test, temp_val, temp_test = load_video_scores()

print("CNN val columns:", cnn_val.columns.tolist())
print("Temporal test columns:", temp_test.columns.tolist())

# If temporal validation file is missing, rebuild validation video scores from available temporal validation predictions
val_candidates = [f for f in os.listdir(RESULT_DIR) if "temporal" in f and "val" in f]
print("Temporal validation related files:", val_candidates)

# %% [markdown]
# ## 7. XAI: Grad-CAM And LIME-Style Explanations
#
# Guideline rule followed here: model decisions are visualized with Grad-CAM and LIME-style superpixel explanations.

# %%
# Step 15: Grad-CAM for fake evidence using Custom CNN
# This version selects a confident fake test frame and explains fake prediction.

def load_image_from_path(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    return img.astype(np.float32) / 255.0

# Pick a correctly predicted fake image with strong fake confidence
fake_candidates = test_frame_df[
    (test_frame_df["true_label"] == 0) &
    (test_frame_df["frame_score_real"] < 0.20)
].sort_values("frame_score_real")

if len(fake_candidates) == 0:
    fake_candidates = test_frame_df[test_frame_df["true_label"] == 0].sort_values("frame_score_real")

sample_path = fake_candidates.iloc[0]["frame_path"]
sample_img = load_image_from_path(sample_path)
sample_label = 0
sample_score_real = float(fake_candidates.iloc[0]["frame_score_real"])

print("Selected image:", sample_path)
print("True label: fake")
print("Real probability:", sample_score_real)
print("Fake probability:", 1 - sample_score_real)

# Use a strong convolution layer
last_conv_layer_name = "block5_sepconv2"
print("Using Grad-CAM layer:", last_conv_layer_name)

def make_gradcam_heatmap_fake(img_array, model, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(img_array, axis=0), training=False)

        # Convert sigmoid probability to logit
        p_real = tf.clip_by_value(predictions[:, 0], 1e-7, 1 - 1e-7)
        logit_real = tf.math.log(p_real / (1.0 - p_real))

        # For fake explanation, explain what pushes prediction AWAY from real
        loss = 1.0 - predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)

    # Stronger normalization so it does not become all blue
    heatmap = heatmap.numpy()
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.percentile(heatmap, 99) + 1e-8)
    heatmap = np.clip(heatmap, 0, 1)

    return heatmap

heatmap = make_gradcam_heatmap_fake(sample_img, custom_cnn, last_conv_layer_name)
heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)

heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

superimposed = np.clip(0.55 * sample_img + 0.45 * heatmap_color, 0, 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(sample_img)
plt.title(f"Original: fake | fake prob={1-sample_score_real:.3f}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap_resized, cmap="jet")
plt.title("Grad-CAM Fake Evidence")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(superimposed)
plt.title("Grad-CAM Overlay")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "gradcam_custom_cnn_fake_evidence.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# Step 15B: LIME-style superpixel explanation
# This implements the key LIME idea: perturb image regions, observe prediction changes, and rank important regions.

try:
    from skimage.segmentation import slic, mark_boundaries
    from sklearn.linear_model import Ridge
    SKIMAGE_AVAILABLE = True
except Exception as e:
    print("LIME-style explanation skipped because skimage/sklearn dependency is missing:", e)
    SKIMAGE_AVAILABLE = False

if SKIMAGE_AVAILABLE:
    img = sample_img.copy()
    segments = slic(img, n_segments=50, compactness=10, sigma=1, start_label=0)
    n_segments = int(segments.max() + 1)
    n_samples = 300

    rng = np.random.default_rng(SEED)
    masks = rng.integers(0, 2, size=(n_samples, n_segments))
    masks[0, :] = 1

    baseline_color = img.mean(axis=(0, 1))
    perturbed_images = []

    for mask in masks:
        perturbed = img.copy()
        for seg_id in range(n_segments):
            if mask[seg_id] == 0:
                perturbed[segments == seg_id] = baseline_color
        perturbed_images.append(perturbed)

    perturbed_images = np.asarray(perturbed_images, dtype=np.float32)
    scores = custom_cnn.predict(perturbed_images, batch_size=32, verbose=0).reshape(-1)

    surrogate = Ridge(alpha=1.0)
    surrogate.fit(masks, scores)
    importances = surrogate.coef_

    top_segments = np.argsort(importances)[-8:]
    explanation_mask = np.isin(segments, top_segments)

    explanation = img.copy()
    explanation[~explanation_mask] = explanation[~explanation_mask] * 0.25

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(mark_boundaries(img, segments))
    plt.title("Superpixels")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(explanation, segments))
    plt.title("LIME-style Important Regions")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "lime_style_custom_cnn.png"), dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 8. Prototype / Deployment Preparation
#
# Guideline rule addressed here: the notebook exports trained models and provides a video prediction function. This is the core logic needed for a future web or mobile prototype.

# %%
# Step 16: Prototype-ready video prediction function

class_names_by_label = {0: "fake", 1: "real"}

def predict_video_file(video_path, threshold=None):
    if threshold is None:
        threshold = best_video_threshold if "best_video_threshold" in globals() else 0.5

    clips = read_multiple_video_clips(video_path, T=T, img_size=IMG_SIZE_CLIP, clips_per_video=CLIPS_PER_VIDEO)
    if len(clips) == 0:
        raise ValueError("Could not read frames from video")

    clip_batch = np.stack([tf.image.resize(clip.astype(np.float32) / 255.0, IMAGE_SIZE).numpy() for clip in clips], axis=0)
    flat_frames = clip_batch.reshape((-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    flat_features = feature_extractor.predict(flat_frames, batch_size=FEATURE_BATCH_SIZE, verbose=0)
    feature_sequences = flat_features.reshape((len(clips), T, -1)).astype(np.float32)
    clip_scores = temporal_model.predict(feature_sequences, verbose=0).reshape(-1)
    video_score = float(np.mean(clip_scores))
    pred_label = int(video_score >= threshold)
    confidence = float(max(video_score, 1.0 - video_score))

    return {
        "video_path": video_path,
        "predicted_label": class_names_by_label[pred_label],
        "real_probability": video_score,
        "fake_probability": 1.0 - video_score,
        "confidence": confidence,
        "clip_scores_real": clip_scores.tolist(),
        "threshold": float(threshold),
        "num_clips_used": int(len(clips)),
    }

# Example usage after training:
# result = predict_video_file(os.path.join(real_video_dir, real_test[0]))
# print(result)

prototype_info = {
    "custom_cnn_model": custom_cnn_path,
   "temporal_model": temporal_model_path,
    "default_threshold": float(best_video_threshold) if "best_video_threshold" in globals() else 0.5,
    "input": "mp4 video file",
    "output": "real/fake label with probability and confidence",
    "function": "predict_video_file(video_path)"
}
with open(os.path.join(RESULT_DIR, "prototype_model_info.json"), "w") as f:
    json.dump(prototype_info, f, indent=2)

print("Prototype inference function ready: predict_video_file(video_path)")

# %% [markdown]
# ## 9. Final Tables And Packaging

# %%
# Step 17: Final thesis result tables, clean final graphs, metadata, and optional external-test template

all_metric_rows = [custom_cnn_metrics]
if "custom_cnn_test_video_metrics" in globals():
    all_metric_rows.append(custom_cnn_test_video_metrics)
all_metric_rows.extend([temporal_clip_metrics, temporal_video_metrics])
if "fusion_video_metrics" in globals():
    all_metric_rows.append(fusion_video_metrics)
if RUN_PRETRAINED_BASELINES and baseline_results:
    all_metric_rows.extend(baseline_results)

metrics_df = pd.DataFrame(all_metric_rows)
metrics_df.to_csv(os.path.join(RESULT_DIR, "final_metrics_table.csv"), index=False)
print("Final metrics:")
print(metrics_df)

print("\nEfficiency:")
print(efficiency_df)

# Clean graph 1: training curves for CNN and temporal model.
def plot_history(history, title, out_name):
    hist = history.history if hasattr(history, "history") else history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    if "accuracy" in hist:
        plt.plot(hist["accuracy"], label="train accuracy")
    if "val_accuracy" in hist:
        plt.plot(hist["val_accuracy"], label="val accuracy")
    if "auc" in hist:
        plt.plot(hist["auc"], label="train AUC")
    if "val_auc" in hist:
        plt.plot(hist["val_auc"], label="val AUC")
    plt.title(title + " - Accuracy/AUC")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    if "loss" in hist:
        plt.plot(hist["loss"], label="train loss")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val loss")
    plt.title(title + " - Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, out_name), dpi=150, bbox_inches="tight")
    plt.show()

plot_history(history_custom_cnn, "Custom CNN", "custom_cnn_training_curves.png")
plot_history(history_temporal, "CNN Attention-BiLSTM", "temporal_training_curves.png")

# Clean graph 2: final comparison bar chart.
plot_metrics = metrics_df.copy()
plt.figure(figsize=(12, 5))
labels = plot_metrics["model"] + "\n(" + plot_metrics["level"] + ")"
x = np.arange(len(plot_metrics))
width = 0.22
plt.bar(x - width, plot_metrics["accuracy"], width, label="Accuracy")
plt.bar(x, plot_metrics["f1"], width, label="F1")
plt.bar(x + width, plot_metrics["auc"], width, label="AUC")
plt.xticks(x, labels, rotation=25, ha="right")
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Final Model Comparison")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "final_model_comparison_bar_chart.png"), dpi=150, bbox_inches="tight")
plt.show()

# Clean graph 3: confusion matrix heatmaps from saved prediction CSVs.
def plot_confusion_from_csv(csv_path, title, out_name, score_col="score_real", label_col="true_label", threshold=0.5):
    df = pd.read_csv(csv_path)
    y_true = df[label_col].astype(int).values
    if "pred_label" in df.columns:
        y_pred = df["pred_label"].astype(int).values
    else:
        y_pred = (df[score_col].values >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xticks([0, 1], ["fake", "real"])
    plt.yticks([0, 1], ["fake", "real"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, out_name), dpi=150, bbox_inches="tight")
    plt.show()

plot_confusion_from_csv(os.path.join(RESULT_DIR, "custom_cnn_test_frame_predictions.csv"), "Custom CNN Confusion Matrix", "custom_cnn_confusion_matrix.png")
plot_confusion_from_csv(os.path.join(RESULT_DIR, "temporal_video_predictions.csv"), "Video-Level CNN-Attention-BiLSTM Confusion Matrix", "temporal_video_confusion_matrix.png", score_col="mean_score_real", threshold=best_video_threshold)
if os.path.exists(os.path.join(RESULT_DIR, "final_fused_video_predictions.csv")):
    plot_confusion_from_csv(os.path.join(RESULT_DIR, "final_fused_video_predictions.csv"), "Final Fused Video Confusion Matrix", "final_fused_video_confusion_matrix.png", score_col="fused_score_real", threshold=fusion_threshold)

experiment_metadata = {
    "seed": SEED,
    "use_full_dataset": USE_FULL_DATASET,
    "n_real": N_REAL,
    "n_fake": N_FAKE,
    "split": "70/20/10 by video before frame/clip extraction",
    "frames_per_video_for_cnn": FRAMES_PER_VIDEO,
    "clip_length_T": T,
    "clips_per_video": CLIPS_PER_VIDEO,
    "frame_image_size": IMAGE_SIZE,
    "clip_image_size": IMG_SIZE_CLIP,
    "augmentation_policy": "training frames only; no validation/test augmentation",
    "selected_video_threshold": float(best_video_threshold) if "best_video_threshold" in globals() else 0.5,
    "fusion_temporal_weight": float(fusion_temporal_weight) if "fusion_temporal_weight" in globals() else None,
    "fusion_threshold": float(fusion_threshold) if "fusion_threshold" in globals() else None,
    "tensorflow_version": tf.__version__,
    "gpu_devices": [str(x) for x in tf.config.list_physical_devices("GPU")],
    "main_limitation": "Model is trained/evaluated on FaceForensics++ C23. Modern AI-generated/diffusion deepfakes require external-dataset testing before making broad claims."
}
with open(os.path.join(RESULT_DIR, "experiment_metadata.json"), "w") as f:
    json.dump(experiment_metadata, f, indent=2)

# Optional external-dataset evaluation template.
# To use it later, point external_real_dir and external_fake_dir to folders of unseen videos.
def evaluate_external_video_folder(external_real_dir, external_fake_dir, max_videos_per_class=None):
    rows = []
    for label_name, label_value, folder in [("real", 1, external_real_dir), ("fake", 0, external_fake_dir)]:
        files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".mp4")])
        if max_videos_per_class is not None:
            files = files[:max_videos_per_class]
        for path in tqdm(files, desc=f"external {label_name}"):
            result = predict_video_file(path)
            rows.append({
                "video_path": path,
                "true_label": label_value,
                "score_real": result["real_probability"],
                "pred_label": 1 if result["predicted_label"] == "real" else 0,
                "confidence": result["confidence"]
            })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        print(classification_report(df["true_label"], df["pred_label"], digits=4, zero_division=0))
        print("External AUC:", roc_auc_score(df["true_label"], df["score_real"]))
        df.to_csv(os.path.join(RESULT_DIR, "external_dataset_predictions.csv"), index=False)
    return df

print("External dataset evaluator is ready: evaluate_external_video_folder(real_dir, fake_dir)")

# %%
# Step 18: Package all thesis outputs

PACKAGE_DIR = os.path.join(PROJECT_ROOT, "packages")
os.makedirs(PACKAGE_DIR, exist_ok=True)
zip_path = os.path.join(PACKAGE_DIR, "deepfake_thesis_guideline_compliant_outputs")

if os.path.exists(zip_path + ".zip"):
    os.remove(zip_path + ".zip")

shutil.make_archive(zip_path, "zip", ARTIFACT_DIR)

print("Packaged outputs:")
print(zip_path + ".zip")
