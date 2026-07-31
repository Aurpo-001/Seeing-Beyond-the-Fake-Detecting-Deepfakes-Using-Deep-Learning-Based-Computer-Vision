# Reproducibility guide

## Environment

The recorded thesis run used TensorFlow 2.21.0 on an NVIDIA RTX 3090. A CUDA-capable GPU is strongly recommended; preprocessing also requires substantial disk space.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure paths:

```bash
export FACEFORENSICS_ROOT=/absolute/path/to/FaceForensics++_C23
export DEEPFAKE_PROJECT_ROOT=/absolute/path/to/deepfake_run
export REBUILD_DERIVED_DATA=1
```

The last variable authorizes regeneration of derived frame and clip directories. Omit it when resuming an existing run.

## Execution order

Run the cleaned notebook from top to bottom in a fresh kernel:

1. Set the deterministic seed and create artifact folders.
2. Validate the dataset layout.
3. Create the video-first 70/20/10 split and save its manifest.
4. Detect/crop faces and extract 32 frames per video.
5. Train and evaluate the Custom CNN.
6. Generate five 24-frame clips per video.
7. Export 512-D CNN features for every clip frame.
8. Train and evaluate the temporal classifier.
9. Tune thresholds and fusion weights on validation videos only.
10. Evaluate the untouched test videos once and export figures/tables.

## What is and is not reproduced here

The public notebook contains the executable experiment logic, but the repository does not include the licensed dataset, trained weights, derived frames/clips, or large prediction tables. The published figures and values are the archived outputs supplied with the thesis.

Validate repository integrity without installing TensorFlow:

```bash
python scripts/validate_notebook.py
```

This checks JSON validity, Python syntax, empty outputs, portable paths, and required research stages.

## Reporting a new run

Record the Git commit, TensorFlow/CUDA versions, GPU, dataset revision, manifest checksum, threshold, fusion weight, and all confusion-matrix counts. Do not compare a frame-random split with the thesis video-first results as if they used the same protocol.
