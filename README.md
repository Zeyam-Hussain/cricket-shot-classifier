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
- [Setting the Epochs (Important)](#setting-the-epochs-important)
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
6. Trains the model using a custom callback that supports interactive checkpoints and automatic learning-rate decay.
7. Plots training/validation loss and accuracy curves and identifies the best epoch.
8. Evaluates the model on the test set (accuracy, F1 score, confusion matrix, classification report).
9. Saves the trained model as a `.keras` file.

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
- **Head:** `BatchNormalization → Dense(256, L1/L2 regularized, ReLU) → Dropout(0.4) → Dense(4, Softmax)`.
- **Optimizer:** Adamax, initial learning rate `0.0001`.
- **Loss:** categorical cross-entropy.

The function also supports `EfficientNetB0`, `B5`, and `B7` by changing `mod_num`, if you want a lighter or heavier backbone.

### 5. Custom training callback — `LR_ASK`
This is the most important custom piece of logic in the notebook:
- After every epoch, it checks the validation loss. If it improved, it stores the current model weights as the new "best weights."
- If validation loss got worse, it automatically reduces the learning rate by a `factor` (default `0.4`) and restores the best weights so far (this behavior is controlled by `dwell=True`).
- At a chosen checkpoint epoch (`ask_epoch`), it pauses training and asks you (via console input) whether to stop (`H`) or continue for a given number of additional epochs.
- When training ends, it automatically reloads the **best-performing weights** recorded during the whole run, regardless of which epoch training actually stopped at.

### 6. Visualization and evaluation
- `tr_plot()` plots training/validation loss and accuracy curves and marks the best epoch on each chart.
- `predictor()` runs the trained model on the test set, prints accuracy and a weighted F1 score, and displays a confusion matrix and classification report.

### 7. Saving the model
The trained model is saved as a `.keras` file with a name that encodes the number of classes, image size, and final F1 score, e.g. `CRICKET-4-(240 X 310)-zeyam-96.50.keras`.

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
2. Run the cells **in order, from top to bottom** — later cells depend on variables created earlier (`train_df`, `classes`, `model`, etc.).
3. When the dataset download cell runs, confirm the printed path matches where the images were saved.
4. When training reaches the `ask_epoch` checkpoint (epoch 10 by default), the notebook will pause and print a prompt in the output. Type:
   - `H` to stop training and keep the best weights found so far, or
   - a number (e.g. `5`) to continue training for that many more epochs before being asked again.
5. After training finishes, run the plotting cell to see the loss/accuracy curves, then the evaluation cell to see test accuracy, F1 score, and the confusion matrix.
6. Run the final cells to save the trained `.keras` model file.

> **Important:** Because the training callback uses Python's `input()` to ask whether to continue, it only works in an **interactive** environment (Jupyter, Colab, Kaggle). Running the notebook as a plain non-interactive script will hang at that prompt unless you set `ask_epoch` to a value greater than or equal to `epochs` (see below) so it never asks.

---

## Configuration Parameters

These are the key variables you can change, and where they appear in the notebook:

| Parameter | Location | Default | Purpose |
|---|---|---|---|
| `img_size` | Cell defining `working_dir` | `(240, 310)` | Height/width images are resized to before training. |
| `batch_size` | Cell calling `make_gens()` | `20` | Number of images per training/validation batch. |
| `mod_num` | `make_model()` call | `3` (EfficientNetB3) | Which EfficientNet backbone to use (`0`, `3`, `5`, or other → `B7`). |
| `lr` | `make_model()` call | `0.0001` | Initial learning rate for the Adamax optimizer. |
| `epochs` | Training cell | `20` | Maximum number of training epochs. |
| `ask_epoch` | Training cell | `10` | Epoch at which training pauses to ask whether to continue. |
| `dwell` | `LR_ASK(...)` call | `True` | If `True`, automatically reduces the learning rate and reloads best weights whenever validation loss worsens. |
| `factor` | `LR_ASK(...)` call | `0.4` | Multiplier applied to the learning rate when `dwell` triggers a reduction. |

---

## Setting the Epochs (Important)

There are two epoch-related settings, and they work together:

- **`epochs`** — the maximum number of epochs the training loop is allowed to run.
- **`ask_epoch`** — the checkpoint at which training pauses so you can decide whether to stop or continue.

### How the "best epoch" is chosen
You do not need to manually guess the best epoch. The `LR_ASK` callback tracks validation loss after every epoch and keeps a copy of the model weights whenever validation loss reaches a new low. When training ends (whether it runs the full `epochs` count or you halt it early), the callback **automatically reloads the best weights it saved**, not just whatever the last epoch produced. The `tr_plot()` function also marks this best epoch visually on the loss/accuracy charts after training, so you can confirm it.

### Recommended epoch settings for this dataset
With ~3,300 training images spread evenly across 4 classes and a **frozen** EfficientNetB3 backbone (only the small classification head is being trained), the model typically converges quickly:

- **Total epochs (`epochs`):** `20–25` is a good starting range. Since the callback restores the best weights automatically, setting this a bit higher than you need is safe and costs little beyond extra training time.
- **Checkpoint (`ask_epoch`):** `8–10` works well for monitoring progress partway through. Because `dwell=True` already auto-decays the learning rate on plateaus and restores best weights, you often don't need to intervene at the checkpoint — entering a number to continue (e.g. `10` more epochs) is usually enough.
- **Best epoch in practice:** for a dataset and model of this size, the lowest validation loss usually appears somewhere between **epoch 8 and epoch 15**; training beyond that mainly risks overfitting the small classification head, which the callback's weight-restoring behavior already guards against.
- **Running unattended (no manual input):** set `ask_epoch = epochs` (e.g. both `20`) so the condition `ask_epoch >= epochs` is met and the callback trains straight through without pausing for input.

If you unfreeze the EfficientNetB3 base for fine-tuning later, expect to need more epochs (typically 25–40) and a lower learning rate, since you would then be training many more parameters.

---

## Model Output

The trained model is saved as:

```
<working_dir>/CRICKET-<num_classes>-(<height> X <width>)-zeyam-<f1_score>.keras
```

Example: `CRICKET-4-(240 X 310)-zeyam-96.50.keras`

By default the save path is `/kaggle/working/...`, which only exists on Kaggle. If running locally or in Colab, change `model_save_loc` to a folder on your own machine or Google Drive.

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
- **Training seems "stuck":** if running outside a notebook, it is likely waiting on the `input()` prompt from `LR_ASK` — set `ask_epoch >= epochs` to avoid this.
- **Out-of-memory errors:** lower `batch_size` (e.g. from 20 to 8 or 16), or switch to a smaller backbone (`mod_num=0` for EfficientNetB0).
- **Different dataset:** if you swap in a different image dataset, update the folder structure to match (one subfolder per class) and adjust `img_size` if aspect ratios differ significantly.
