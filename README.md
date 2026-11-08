# HandGestureAI-ML

An AI-powered hand gesture recognition system developed using Python, TensorFlow, and deep learning. This project explores custom neural network development, computer vision, hand landmark detection, gesture classification, and TensorFlow Lite deployment.

## Overview

This project focuses on building a custom machine learning pipeline for real-time hand gesture recognition using two independently developed deep learning models.

**Model 1 – Hand Landmark Detection**

A custom heatmap-based deep learning model that detects 21 hand landmarks and predicts their coordinates from an input image.

**Model 2 – Hand Gesture Classification**

A custom neural network that receives 42 normalized landmark coordinate values (21 X,Y points) and classifies the corresponding hand gesture.

Both models are developed using TensorFlow and Keras and converted into TensorFlow Lite format for efficient inference.

## Key Features

- Custom heatmap-based hand landmark detection
- Detection of 21 hand landmarks
- Landmark coordinate decoding and normalization
- 42-feature coordinate-based gesture classification
- Custom neural network development and training
- Dataset preprocessing and batch processing
- Training and validation
- Performance evaluation using accuracy, loss, and confusion matrices
- TensorFlow Lite model conversion
- Keras and TensorFlow Lite prediction consistency validation

## Technologies & Tools

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV
- Scikit-learn
- Matplotlib
- Pillow
- TensorFlow Lite
- Git & GitHub

## Machine Learning Workflow

1. **Image Processing:** Prepare hand images for model input.
2. **Landmark Detection:** Predict 21 hand landmarks using the custom-trained landmark model.
3. **Feature Extraction:** Decode and normalize landmark coordinates into 42 numerical features.
4. **Gesture Classification:** Process extracted features using a custom neural network.
5. **Model Training:** Train and validate both models using their respective datasets.
6. **Model Evaluation:** Analyse model performance using accuracy, loss, and confusion matrices.
7. **Model Conversion:** Convert trained Keras models into TensorFlow Lite format.
8. **Conversion Validation:** Compare Keras and TensorFlow Lite predictions to verify conversion consistency.

## Dataset Requirements

**Important:** Dataset images and original training data are not included in this GitHub repository. They must be downloaded separately and prepared locally before running the training scripts.

### 1. FreiHAND Dataset – Hand Landmark Detection

Download: https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html

Used for training the custom 21-point hand landmark detection model.

Required data includes:
- RGB training images
- `training_xyz.json`
- `training_K.json`

The original FreiHAND training release contains 130,240 RGB image variants.

### 2. HG14 HandGesture14 Dataset – Gesture Classification

Download: https://www.kaggle.com/datasets/gulerosman/hg14-handgesture14-dataset

Used for gesture image preprocessing and classifier training.

The dataset contains 14 gesture categories. The images must be extracted and organized into their respective gesture folders.

### Dataset Preparation

After downloading and preparing the datasets, organize them using the expected structure:

```text
gesture_ml/
│
├── data/
│   ├── data_set/
│   │   ├── freihand/
│   │   │   └── rgb/
│   │   │
│   │   └── HG14-Hand Gesture/
│   │       ├── Gesture_0/
│   │       ├── Gesture_1/
│   │       ├── Gesture_2/
│   │       └── ...
│   │
│   └── gestures_hg14.csv
│
├── model/
├── output/
├── script/
├── src/
│   └── common/
│
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** The dataset directory is intentionally excluded from GitHub because of its size. Users must download the datasets and prepare the required files locally.

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Dhanush-Pb/HandGestureAI-ML.git
cd HandGestureAI-ML
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Download and prepare datasets

Download the FreiHAND and HG14 datasets using the links above. Extract them and organize the required files under the local `data/` directory.

## Model Training

Before training, ensure that the required datasets and annotation files are available locally.

**Landmark Detection Model**

```bash
python script/train_landmarks.py
```

**Gesture Classification Model**

Run the corresponding gesture preprocessing script to generate `gestures_hg14.csv`, then execute the gesture training script.

Check the script filenames and dataset paths in your local project before execution.

## Model Deployment

The trained Keras models are converted into TensorFlow Lite format for lightweight inference.

The conversion workflow also compares predictions from the original Keras models with their TensorFlow Lite versions to validate conversion consistency.

## Repository Notes

- Original dataset images are not uploaded.
- Downloaded datasets are excluded using `.gitignore`.
- Generated training backups and temporary files are excluded.
- Python dependencies are listed in `requirements.txt`.
- Model training requires the original datasets to be prepared locally.

## Author

**Dhanush PB**

GitHub: [@Dhanush-Pb](https://github.com/Dhanush-Pb)

---

*Exploring Machine Learning, Deep Learning, and Computer Vision through practical implementation.*