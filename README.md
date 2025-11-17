# CT-Triage: Brain Stroke Classification from CT Scans

This repository contains the code for a **CT triage system** that classifies brain CT slices into stroke-related categories (e.g. *Normal*, *Ischemia*, *Bleeding*).  
The goal is to provide a **baseline deep learning pipeline** for training, evaluating and comparing convolutional neural networks for medical triage.

The project is implemented in PyTorch and is designed to run both locally and in Google Colab.

---

## Main Features

- Supervised image classification on brain CT images.
- Multiple backbones:
  - ResNet-18
  - ResNet-50
  - DenseNet-121
  - EfficientNet-B0
- Training script that:
  - Loads data from an organized folder structure.
  - Trains a chosen backbone.
  - Tracks metrics (accuracy, precision, recall, F1, AUC).
  - Saves the **best checkpoint** based on F1-score.
  - Logs all epochs to `training_log.csv`.
- Evaluation script for:
  - Loading saved checkpoints.
  - Computing metrics on internal / external test sets.
- Ready to use with **Google Drive + Colab** (no dataset stored in this repo).

---

## Repository Structure

```text
ct_triage/
├─ src/
│  ├─ dataset.py          # Dataset and dataloader utilities
│  ├─ models.py           # Model builders (ResNet, DenseNet, EfficientNet, etc.)
│  ├─ train_baseline.py       # Trains a baseline deep-learning model using a chosen backbone and saves the best checkpoint and training logs.
│  ├─ .gitkeep
│  └─ utils.py            # Helper functions
├─ notebooks/
│  ├─ .gitkeep
│  └─ ct_triage.ipynb   # Main Colab notebook (exploration / experiments)
├─ requirements.txt       # Python dependencies
├─ .gitignore
└─ README.md

```

As previously stated, there is no dataset included in this repository. 
The expected folder structure is:
Brain_Stroke_CT_Dataset/
├── Normal/
│  ├── PNG/
├── Ischemia/
│  ├── PNG/
├── Bleeding/
│  ├── PNG/
└──External_Test/
│  ├── PNG/
│  ├── MASKS/
│  └── labels.csv 


REQUIRED: Place your dataset under Google Drive:
/content/drive/MyDrive/ct_triage_data/Brain_Stroke_CT_Dataset/


---

## Running in Google Colab

### 1. Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
### 2. Move into your proyect folder
```python
%cd /content/drive/MyDrive/ct_triage/ct_triage
```
### 3. Install dependencies
```python
!pip install -r requirements.txt
```
---

## Training and evaluation

To train the baseline model, simply run the provided training notebook:

** `notebooks/ct_triage.ipynb`**

This notebook handles the full training pipeline, including:

- loads and preprocesses the dataset  
- trains multiple backbones (e.g., ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0)  
- saves the best checkpoint and training logs under `runs/`

It also performs evaluation by:

- computing accuracy, precision, recall, F1 and AUC for **all trained models**  
- selecting the best model and generating a **confusion matrix** for that best model

Just open the notebook in Google Colab or Jupyter, follow the cells in order, and the model will be trained automatically.

---





