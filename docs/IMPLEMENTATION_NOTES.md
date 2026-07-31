# Implementation notes

## Public-copy changes

The original notebook remains unchanged outside this repository. The public copy only applies portability and safety changes: clearing outputs, replacing machine paths, initializing optional-baseline state, removing temporary inspection cells, and guarding derived-data deletion.

## Temporal architecture label

The thesis and several archived figures use the label **CNN-Attention-BiLSTM** for the temporal branch. The executable final notebook function `build_bilstm_temporal_model` uses a bidirectional LSTM followed by global average and max pooling, dense layers, and dropout; it does not instantiate a `MultiHeadAttention` layer.

For reproducibility, this repository preserves the executable model rather than silently adding a new attention block. Adding explicit temporal attention would define a new experiment and requires retraining, fresh threshold selection, and a new test evaluation.

## Metric convention

Keras directory ordering maps `fake = 0` and `real = 1`. Reported precision and recall therefore treat `real` as the positive class unless a result explicitly states otherwise. Confusion matrices use rows as true labels and columns as predicted labels in `[fake, real]` order.
