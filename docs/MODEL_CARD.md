# Model card

## Model overview

The framework classifies facial videos as real or fake through complementary spatial and temporal evidence. A lightweight Custom CNN handles individual frames and supplies 512-dimensional features to a video branch. A validation-tuned fusion rule combines spatial and temporal probabilities.

| Component | Description |
| --- | --- |
| Spatial backbone | Scratch-built CNN with squeeze-and-excitation blocks |
| Spatial parameters | 511,253 |
| Frame input | RGB, 224 x 224 |
| Temporal input | 24 ordered frame features, 512 values each |
| Temporal classifier in notebook | Bidirectional LSTM with global average/max pooling and dense layers |
| Output | Probability of the `real` class; fake probability is `1 - p(real)` |

## Training data

The final pipeline uses FaceForensics++ C23 videos, with `original` as real and `Deepfakes` as manipulated. The research split contains 70% training, 20% validation, and 10% test videos per class. Partitioning occurs before frames or clips are generated.

The repository does not distribute the dataset, extracted faces, trained weights, or per-video prediction files.

## Performance

| Model | Accuracy | AUC | Evaluation unit |
| --- | ---: | ---: | --- |
| Custom CNN | 90.06% | 96.44% | 6,400 test frames |
| Temporal branch | 92.00% | 96.45% | 200 test videos |
| Fusion framework | 93.50% | 97.99% | 200 test videos |

Metrics are archival thesis results, not results recomputed by this repository packaging process.

## Intended use

- Academic research on deepfake detection.
- Reproduction and extension of video-first evaluation protocols.
- Study of lightweight spatial representations and temporal fusion.
- Prototyping media-forensics decision support with human oversight.

## Out-of-scope use

- Automated accusations, disciplinary action, employment decisions, or legal judgments.
- Identity verification without independent safeguards.
- Claims of broad real-world accuracy based only on the reported benchmark.
- Detection of audio, full-body, or multimodal synthetic content.

## Risks and limitations

- A single primary benchmark does not measure cross-dataset generalization.
- Compression, resizing, camera pipelines, demographics, and generator families may cause performance shifts.
- False positives can harm people whose authentic media is incorrectly flagged.
- Explainability maps show model sensitivity, not causal proof of manipulation.
- Thresholds were selected on validation data and must be recalibrated for a new deployment domain.

## Recommended safeguards

Use calibrated thresholds, preserve audit logs, evaluate demographic and source-domain performance, include human review, and pair detector output with provenance and contextual evidence.
