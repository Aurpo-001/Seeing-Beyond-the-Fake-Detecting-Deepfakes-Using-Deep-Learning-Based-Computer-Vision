# Dataset and preparation

## FaceForensics++ C23

The final study uses the C23 compressed version of FaceForensics++. Obtain access from the official dataset maintainers and follow their terms. This repository does not contain source videos or extracted frames.

Expected local layout:

```text
FaceForensics++_C23/
├── original/
│   └── *.mp4
└── Deepfakes/
    └── *.mp4
```

## Leakage-resistant split

The random seed is 42. Real and fake filename lists are independently shuffled, then split by source video:

| Partition | Real videos | Fake videos | Share |
| --- | ---: | ---: | ---: |
| Train | 700 | 700 | 70% |
| Validation | 200 | 200 | 20% |
| Test | 100 | 100 | 10% |

Frames and clips are generated only after this split. A saved manifest records every filename assigned to each partition.

## Derived frame data

- 32 evenly spaced frames per source video.
- Largest detected frontal face with margin; center-crop fallback.
- Resize to 224 x 224 pixels.
- Augmentation only in the training pipeline.
- Final test set: 3,200 real and 3,200 fake frames.

## Derived clip data

- Five clips per source video.
- 24 ordered frames per clip.
- Initial clip resize to 160 x 160, then 224 x 224 before CNN feature extraction.
- Short videos are padded by repeating the final available frame.

Generated frames, clips, features, weights, manifests, and predictions are ignored by Git and should remain local.
