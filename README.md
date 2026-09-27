# Cricket Shot Image Classification

A transfer-learning image classifier built with **TensorFlow / Keras** and **EfficientNetB3** that identifies four types of cricket batting shots from images: **Drive**, **Leg Glance / Flick**, **Pull Shot**, and **Sweep**.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [How the Code Works](#how-the-code-works)
- [Requirements](#requirements)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Configuration Parameters](#configuration-parameters)
- [Training Strategy and Epochs](#training-strategy-and-epochs)
- [Model Output](#model-output)
- [Evaluation](#evaluation)
- [Notes and Troubleshooting](#notes-and-troubleshooting)

---

## Overview

The notebook (`image-classification-cricketer.ipynb`) trains an image classification model in the following stages:

1. Downloads the dataset from Kaggle.
2. Builds a labeled DataFrame and splits it into train, validation, and test sets.
3. Balances the training set by trimming every class to an equal number of samples.
4. Creates Keras `ImageDataGenerator` pipelines for training, validation, and testing.
5. Builds a transfer-learning model on top of a frozen **EfficientNetB3** backbone.
6. Trains the model using `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau`, all monitoring **validation accuracy**.
7. Reloads the best-performing checkpoint saved during training.
8. Plots training/validation loss and accuracy curves.
9. Evaluates the model on the test set (accuracy, F1 score, confusion matrix, classification report).
10. Saves the final model as a `.keras` file.

---

## Dataset

- **Source:** [`aneesh10/cricket-shot-dataset`](https://www.kaggle.com/datasets/aneesh10/cricket-shot-dataset) on Kaggle.
- **Classes (4):** `drive`, `legglance-flick`, `pullshot`, `sweep`.
- **Split:** 70% train / 15% validation / 15% test (stratified by class).
- **Balancing:** Each training class is trimmed to **784 images** so all classes are equal in size before training.
- The dataset downloads automatically the first time the notebook runs; you do not need to download it manually unless you want to.

---

## How the Code Works

### 1. Data loading and splitting
`make_dataframes()` walks the dataset folder, builds a DataFrame of file paths and labels, and creates stratified train/validation/test splits. It also samples a subset of images to estimate average image dimensions.

### 2. Class balancing
`trim()` caps every class at 784 images so the model is not biased toward a majority class. An additional `balance()` function (image augmentation via rotation, shift, zoom, and horizontal flip) is defined in the notebook but is **not called** in the current pipeline, since trimming already produces an even class distribution.

### 3. Data generators
`make_gens()` creates three Keras `ImageDataGenerator` flows:
- **Train generator:** shuffled, batch size 20.
- **Validation generator:** unshuffled, batch size 20.
- **Test generator:** batch size is calculated automatically so the full test set is covered in exact, non-overlapping batches.

Images are resized to **240 × 310** pixels.

### 4. Model architecture
`make_model()` builds a transfer-learning classifier:
- **Backbone:** EfficientNetB3 (`mod_num=3`), pretrained on ImageNet, with the top layers removed and the base **frozen** (not trainable).
- **Head:** `BatchNormalization → Dense(256, L2-regularized, ReLU) → Dropout(0.4) → Dense(4, Softmax)`.
- **Optimizer:** Adamax, initial learning rate `0.0001`.
- **Loss:** categorical cross-entropy.

The regularization on the Dense(256) layer was reduced from the original setup (L2 `0.016` plus L1 penalties on activity and bias) down to a single **L2 penalty of `0.001`**. The original settings pushed the raw loss value so low it no longer tracked classification quality, causing the "best epoch" to drift toward the very last epoch instead of where the model actually generalized best.

The function also supports `EfficientNetB0`, `B5`, and `B7` by changing `mod_num`, if you want a lighter or heavier backbone.

### 5. Training callbacks
Training now uses three built-in Keras callbacks, all monitoring **`val_accuracy`** rather than loss:

- **`ModelCheckpoint`** — saves the model to `best_model.keras` every time validation accuracy reaches a new high, so the best checkpoint is always on disk even if training is interrupted.
- **`EarlyStopping`** — stops training if validation accuracy hasn't improved for 5 epochs in a row, and automatically restores the model's weights from the best epoch seen (`restore_best_weights=True`).
- **`ReduceLROnPlateau`** — cuts the learning rate by a factor of 0.4 if validation accuracy hasn't improved for 2 epochs, letting training keep refining instead of stalling.

The notebook still contains an earlier custom callback, `LR_ASK`, which paused training with an interactive prompt and tracked the best epoch by validation **loss**. It is left in the notebook but is **no longer used** in the training cell, since it required manual input and its loss-based checkpoint selection was unreliable (see above).

### 6. Loading the best checkpoint
Immediately after training, the notebook runs:
```python
model = keras.models.load_model("best_model.keras")
```
This guarantees that all downstream plotting, evaluation, and saving use the actual best-accuracy checkpoint, even if the notebook is re-run out of order or the kernel is restarted later.

### 7. Visualization and evaluation
- `tr_plot()` plots training/validation loss and accuracy curves and marks the best epoch on each chart.
- `predictor()` runs the trained model on the test set, prints accuracy and a weighted F1 score, and displays a confusion matrix and classification report.

### 8. Saving the model
The trained model is saved as a `.keras` file with a name that encodes the number of classes, image size, and final F1 score, e.g. `CRICKET-4-(240 X 310)-zeyam-96.50.keras`. This cell depends on `classes`, `img_size`, `f1score`, and `model` all already existing in the session, so `predictor(test_gen)` must be run before it.

---

## Requirements

- **Python:** 3.10–3.12 (the original notebook was run on Python 3.12).
- **Recommended:** a GPU (NVIDIA CUDA-compatible) for reasonable training time. EfficientNetB3 will run on CPU but much more slowly.

### Python packages

```
kagglehub
pandas
numpy
opencv-python
matplotlib
seaborn
scikit-learn
tensorflow>=2.12
tqdm
```

Install everything with:

```bash
pip install kagglehub pandas numpy opencv-python matplotlib seaborn scikit-learn tensorflow tqdm
```

If you have a compatible GPU, install `tensorflow` with GPU support according to the [official TensorFlow install guide](https://www.tensorflow.org/install) for your CUDA/cuDNN version.

---

## Setup

### 1. Kaggle access (for automatic dataset download)
The notebook uses `kagglehub.dataset_download()`, which needs a Kaggle account and API token the first time it runs.

1. Create a Kaggle account if you don't have one.
2. Go to **Kaggle → Account → Create New API Token**. This downloads a `kaggle.json` file.
3. Place it at:
   - Linux/macOS: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
4. Alternatively, set these environment variables instead of using the file:
   ```bash
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   ```

If you'd rather not set up Kaggle credentials, download the dataset manually from the [dataset page](https://www.kaggle.com/datasets/aneesh10/cricket-shot-dataset), extract it, and replace the `kagglehub.dataset_download(...)` line with the local folder path.

### 2. Environment
Create a virtual environment (recommended) before installing packages:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

(You can save the package list above into a `requirements.txt` file.)

---

## How to Run

1. Open the notebook in Jupyter Lab/Notebook, Google Colab, or Kaggle Notebooks.
2. Run the cells **in order, from top to bottom** — later cells depend on variables created earlier (`train_df`, `classes`, `model`, etc.). If you edit an earlier cell, re-run it and every cell after it.
3. When the dataset download cell runs, confirm the printed path matches where the images were saved.
4. Run the training cell. Training now runs **straight through without pausing for input** — there is no interactive prompt, so it's safe to run non-interactively (e.g. as a script or in Kaggle's "Save & Run All").
5. Run the checkpoint-loading cell (`model = keras.models.load_model("best_model.keras")`) so the rest of the notebook uses the correct best-accuracy model.
6. Run the plotting cell to see the loss/accuracy curves, then the evaluation cell to see test accuracy, F1 score, and the confusion matrix.
7. Run the final cells to save the trained `.keras` model file.

> **On Kaggle:** `best_model.keras` and the final named `.keras` file are both saved under `/kaggle/working/`. Files there only persist for the current session unless you **Save Version** (commit the notebook) — download them or commit right after training if you want to keep them.

---

## Configuration Parameters

| Parameter | Location | Default | Purpose |
|---|---|---|---|
| `img_size` | Cell defining `working_dir` | `(240, 310)` | Height/width images are resized to before training. |
| `batch_size` | Cell calling `make_gens()` | `20` | Number of images per training/validation batch. |
| `mod_num` | `make_model()` call | `3` (EfficientNetB3) | Which EfficientNet backbone to use (`0`, `3`, `5`, or other → `B7`). |
| `lr` | `make_model()` call | `0.0001` | Initial learning rate for the Adamax optimizer. |
| `epochs` | Training cell | `30` | Maximum number of training epochs. |
| `monitor` (all three callbacks) | Training cell | `val_accuracy` | Metric used to decide the best checkpoint, when to stop, and when to reduce the learning rate. |
| `patience` (`EarlyStopping`) | Training cell | `5` | Epochs to wait without improvement before stopping. |
| `patience` (`ReduceLROnPlateau`) | Training cell | `2` | Epochs to wait without improvement before cutting the learning rate. |
| `factor` (`ReduceLROnPlateau`) | Training cell | `0.4` | Multiplier applied to the learning rate when it's reduced. |

---

## Training Strategy and Epochs

`epochs = 30` is only an upper bound, not a fixed training length. The actual stopping point and the model that gets kept are both decided automatically:

- **Best checkpoint:** `ModelCheckpoint` saves `best_model.keras` every time validation accuracy improves.
- **Stopping point:** `EarlyStopping` halts training once validation accuracy stops improving for 5 straight epochs, and restores the best-epoch weights into `model` in memory.
- **Learning rate:** `ReduceLROnPlateau` shrinks the learning rate whenever validation accuracy stalls for 2 epochs, helping the model keep improving in smaller steps instead of stalling out early or overshooting.

Because all three callbacks watch **accuracy** instead of raw loss, the selected "best" epoch now reflects actual classification performance. This matters here specifically because the model's loss includes a regularization term that can keep falling even when accuracy plateaus or drops, which previously caused the wrong epoch to be selected.

You generally do not need to change `epochs` upward, since `EarlyStopping` will stop training on its own once it stops helping. Lowering `epochs` only matters if you want a hard time/compute limit regardless of whether training has plateaued.

---

## Model Output

Two model files are produced:

1. **`best_model.keras`** — written automatically during training by `ModelCheckpoint`, representing the epoch with the highest validation accuracy.
2. **The final named export**, produced after evaluation:
   ```
   <working_dir>/CRICKET-<num_classes>-(<height> X <width>)-zeyam-<f1_score>.keras
   ```
   Example: `CRICKET-4-(240 X 310)-zeyam-96.50.keras`

By default the final save path is `/kaggle/working/...`, which only exists on Kaggle. If running locally or in Colab, change `model_save_loc` to a folder on your own machine or Google Drive.

---

## Evaluation

After training, the notebook reports:
- **Test accuracy** — percentage of correctly classified test images.
- **Weighted F1 score** — accounts for any class imbalance in the test set.
- **Confusion matrix** — visual breakdown of predicted vs. actual shot types.
- **Classification report** — precision, recall, and F1 score per class.

---

## Notes and Troubleshooting

- **Slow training / no GPU:** EfficientNetB3 is a fairly large backbone. On CPU-only machines, expect significantly longer epoch times; consider Google Colab or Kaggle Notebooks, which provide free GPU access.
- **Kaggle authentication errors:** double-check `kaggle.json` placement or the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables.
- **`NameError` on the save cell:** the final save cell needs `classes`, `img_size`, `f1score`, and `model` to already exist. Run the checkpoint-loading cell and `predictor(test_gen)` before it.
- **Interrupted training cell:** stopping a cell mid-run does not reset `model`'s weights, but it does prevent `history` from being reassigned. To resume, call `model.fit(...)` again with `initial_epoch` set to the number of epochs already completed, and merge the two `History` objects before plotting.
- **Stale plots:** `tr_plot(history, 0)` always plots whatever `history` currently holds. If a training cell fails or is interrupted before finishing, `history` still refers to the previous successful run. Restart the kernel and re-run all cells top to bottom if the plots look outdated.
- **Out-of-memory errors:** lower `batch_size` (e.g. from 20 to 8 or 16), or switch to a smaller backbone (`mod_num=0` for EfficientNetB0).
- **Different dataset:** if you swap in a different image dataset, update the folder structure to match (one subfolder per class) and adjust `img_size` if aspect ratios differ significantly.