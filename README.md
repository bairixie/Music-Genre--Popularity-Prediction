# Music Genre and Popularity Prediction

A comprehensive machine learning project that predicts music genre classification and popularity regression using various machine learning algorithms and deep learning techniques. This project implements both single-task and multi-task learning approaches to solve two related problems simultaneously.

## Abstract

This project studies how well machine learning models can predict both the genre and popularity of songs using only audio-derived features. From the point of view of a streaming platform, this corresponds to a **cold-start scenario**, where a brand-new track has no user history and the system must rely purely on how the song sounds. Using a real-world dataset of song-level descriptors (danceability, energy, valence, tempo, loudness, etc.), we benchmark standard classification and regression models and analyze how different feature subsets affect performance.

Tree-based algorithms such as Random Forest and XGBoost consistently outperform linear models for both genre and popularity, indicating strong nonlinear relationships between musical characteristics and listener behavior. We then implement a multi-task neural network that jointly predicts genre and popularity from a shared representation, and compare it against the single-task baselines to assess when multi-task learning provides benefits in this audio-feature setting.

## Table of Contents

- [Abstract](#abstract)
- [Motivation](#motivation)
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline](#pipeline)
- [Models](#models)
- [Results](#results)
- [File Descriptions](#file-descriptions)

## Motivation

Music streaming platforms like Spotify and Apple Music must recommend newly released tracks before any listening history is available. This **"cold-start" setting** motivates methods that infer both the semantic category of a song and its potential appeal directly from acoustic features. Being able to estimate genre and likely popularity from audio alone is useful to:

- **Platforms**: Automatic playlisting and promotion of new tracks
- **Artists and Producers**: Feedback about how musical choices may affect reach
- **Music Discovery**: Enabling better recommendations for users

This work studies two related prediction tasks based solely on audio-derived features: (i) multiclass genre classification and (ii) regression of a popularity score in [0,100]. Our project is primarily an application and benchmarking study on an existing dataset, complemented with a multi-task learning (MTL) component.

## Overview

This project addresses two key problems in music analysis:

1. **Music Genre Classification**: Multi-class classification task to predict the genre of a song from audio features
2. **Popularity Prediction**: Regression task to predict the popularity score of a song

The project explores various machine learning approaches including traditional ML models, ensemble methods, and deep neural networks with multi-task learning capabilities. We first establish strong single-task baselines using logistic regression, SVM, KNN, Random Forest, Gradient Boosting, XGBoost, and a feed-forward neural network. We then train a shared multi-task neural network that jointly predicts genre and popularity and compare it to the best single-task models.

## Project Structure

```
Music-Genre--Popularity-Prediction/
├── data_precessing/                         # Data preprocessing scripts
│   ├── datacleaning.py                      # Data cleaning and preprocessing
│   ├── datapreprocessing.py                 # Genre one-hot encoding
│   ├── datastandardization.py               # Data standardization for both tasks
│   ├── datastandardization_classification.py # PCA-based preprocessing for classification
│   ├── datastandardization_regression.py    # PCA-based preprocessing for regression
│   └── mergedataset.py                      # Merge datasets for multi-task learning
├── model_training_benchmarking/              # Model training notebooks
│   ├── genre_baseline_prediction.ipynb      # Genre classification models
│   ├── popularity_baseline_prediction.ipynb # Popularity regression models
│   └── mult_task_learning.ipynb             # Multi-task neural network
├── raw_dataset/                             # Raw data files
│   └── musicData.csv                        # Original dataset
└── README.md                                 # This file
```

## Features

- **Comprehensive Data Preprocessing**: Handles missing values, encodes categorical features, and normalizes data
- **Feature Engineering**: Includes key and mode encoding, tempo normalization, and duration imputation
- **Multiple Model Architectures**: 
  - Traditional ML: Logistic Regression, SVM, KNN, Random Forest
  - Ensemble Methods: XGBoost, Gradient Boosting
  - Deep Learning: Multi-layer Perceptron (MLP), Multi-task Neural Networks
- **Dimensionality Reduction**: PCA-based feature reduction for both tasks
- **Multi-task Learning**: Joint learning framework for simultaneous genre and popularity prediction
- **Comprehensive Evaluation**: Multiple metrics for classification and regression tasks

## Dataset

The dataset (`raw_dataset/musicData.csv`) contains music tracks with the following features:

- **Audio Features**: `acousticness`, `danceability`, `energy`, `valence`, `tempo`, `duration_ms`, `instrumentalness`, `liveness`, `loudness`, `speechiness`
- **Musical Features**: `key`, `mode` (encoded as `key_code` and `mode_code`)
- **Target Variables**: 
  - `music_genre`: Genre classification label (multiple genres)
  - `popularity`: Numerical popularity score (0-100)

**Dataset Statistics**:
- Total samples: ~50,000 tracks
- Training set: 35,000 samples (70%)
- Validation set: 7,500 samples (15%)
- Test set: 7,500 samples (15%)

## Installation

### Prerequisites

- Python 3.7+
- pip or conda

### Required Packages

```bash
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn joblib
```

Or install from requirements (if available):

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Data Cleaning

Clean and preprocess the raw data:

```bash
cd data_precessing
python datacleaning.py
```

This script:
- Removes unnecessary columns (`artist_name`, `track_name`, `obtained_date`, `instance_id`)
- Handles missing values in `duration_ms` and `tempo` using genre-based imputation
- Encodes `key` and `mode` features numerically
- Outputs: `musicData_clean.csv`

### Step 2: Genre One-Hot Encoding

Encode music genres as one-hot vectors:

```bash
cd data_precessing
python datapreprocessing.py
```

This script:
- Converts genre labels to one-hot encoded format
- Creates genre mapping file
- Outputs: `musicData_genreOneHot.csv`, `music_genre_mapping.csv`

### Step 3: Data Standardization

Prepare standardized datasets for both tasks:

```bash
cd data_precessing
python datastandardization.py
```

This script:
- Splits data into train/validation/test sets (70/15/15) with stratified sampling
- Standardizes features using StandardScaler
- Creates separate datasets for classification and regression tasks
- Saves scalers and imputers for inference
- Outputs: 
  - `train_classification_std.csv`, `val_classification_std.csv`, `test_classification_std.csv`
  - `train_regression_std.csv`, `val_regression_std.csv`, `test_regression_std.csv`
  - `scaler_classification.pkl`, `scaler_regression.pkl`
  - `split_indices.npz`

### Step 4 (Optional): PCA-based Preprocessing

For dimensionality reduction:

```bash
cd data_precessing
# Classification task with PCA
python datastandardization_classification.py

# Regression task with PCA
python datastandardization_regression.py
```

### Step 5: Multi-task Dataset Preparation

Merge classification and regression datasets for multi-task learning:

```bash
cd data_precessing
python mergedataset.py
```

Outputs: `train_multitask.csv`, `val_multitask.csv`, `test_multitask.csv`

### Step 6: Model Training

#### Genre Classification

Open and run `model_training_benchmarking/genre_baseline_prediction.ipynb` to train and evaluate classification models:

- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest
- XGBoost
- Neural Network (MLP)

#### Popularity Prediction

Open and run `model_training_benchmarking/popularity_baseline_prediction.ipynb` to train and evaluate regression models:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

#### Multi-task Learning

Open and run `model_training_benchmarking/mult_task_learning.ipynb` to train a deep neural network that simultaneously predicts both genre and popularity.

## Pipeline

The complete data processing pipeline:

```
Raw Data (raw_dataset/musicData.csv)
    ↓
[data_precessing/datacleaning.py]
    ↓
Cleaned Data (musicData_clean.csv)
    ↓
[data_precessing/datapreprocessing.py]
    ↓
One-Hot Encoded Data (musicData_genreOneHot.csv)
    ↓
[data_precessing/datastandardization.py]
    ↓
Standardized Datasets (train/val/test for classification & regression)
    ↓
[data_precessing/mergedataset.py] (optional)
    ↓
Multi-task Datasets (train/val/test_multitask.csv)
    ↓
[model_training_benchmarking/: Model Training & Evaluation]
```

## Models

### Classification Models (Genre Prediction)

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear classifier with multi-class support |
| **SVM** | Support Vector Machine with RBF kernel |
| **KNN** | K-Nearest Neighbors (k=5) |
| **Random Forest** | Ensemble of 300 decision trees |
| **XGBoost** | Gradient boosting with 300 estimators |
| **MLP** | Multi-layer Perceptron (128, 64 hidden units) |

### Regression Models (Popularity Prediction)

| Model | Description |
|-------|-------------|
| **Linear Regression** | Standard linear regression |
| **Ridge** | L2-regularized linear regression (α=1.0) |
| **Lasso** | L1-regularized linear regression (α=0.001) |
| **Random Forest** | Ensemble of 1000 trees |
| **Gradient Boosting** | 100 estimators, learning rate 0.1 |
| **XGBoost** | 1000 estimators, learning rate 0.05 |

### Multi-task Neural Network

- **Architecture**: Shared layers (256 → 128) with task-specific heads
- **Shared Layers**: Dense layers with BatchNormalization and Dropout (0.3)
- **Genre Head**: Dense layer with softmax activation
- **Popularity Head**: Dense layer with linear activation
- **Loss Function**: Combined categorical crossentropy (genre) + MSE (popularity)

## Results

### Genre Classification Performance

Best performing models (on test set with all features):

| Model | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) |
|-------|----------|-------------------|----------------|------------------|
| **XGBoost** | 0.548 | 0.543 | 0.548 | 0.544 |
| **Random Forest** | 0.479 | 0.469 | 0.479 | 0.472 |
| **SVM** | 0.456 | 0.447 | 0.456 | 0.443 |
| **MLP** | 0.449 | 0.440 | 0.449 | 0.438 |
| **Logistic Regression** | 0.393 | 0.387 | 0.393 | 0.384 |
| **KNN** | 0.367 | 0.367 | 0.367 | 0.363 |

### Popularity Prediction Performance

Best performing models (on validation set with all features):

| Model | RMSE | MAE | R² | Spearman Correlation |
|-------|------|-----|-----|---------------------|
| **Random Forest** | 11.83 | - | - | - |
| **XGBoost** | 11.91 | - | - | - |
| **Gradient Boosting** | 12.53 | - | - | - |

*Note: Complete metrics available in the notebooks*

## File Descriptions

### Data Processing Scripts (`data_precessing/`)

- **`datacleaning.py`**: 
  - Removes irrelevant columns
  - Handles missing values with genre-based imputation
  - Encodes musical keys and modes
  - Outputs cleaned dataset

- **`datapreprocessing.py`**: 
  - Performs one-hot encoding for music genres
  - Creates genre mapping file

- **`datastandardization.py`**: 
  - Splits data into train/val/test sets with stratification
  - Standardizes features separately for classification and regression
  - Saves preprocessing objects for inference

- **`datastandardization_classification.py`**: 
  - Applies PCA (95% variance) for classification task
  - Reduces dimensionality while preserving information

- **`datastandardization_regression.py`**: 
  - Applies PCA (95% variance) for regression task
  - Optimized feature set for popularity prediction

- **`mergedataset.py`**: 
  - Combines classification and regression datasets
  - Prepares data for multi-task learning

### Model Training Notebooks (`model_training_benchmarking/`)

- **`genre_baseline_prediction.ipynb`**: 
  - Comprehensive genre classification experiments
  - Multiple feature sets (All, Core_Audio, Rhythm)
  - Model comparison and evaluation
  - Visualization of results

- **`popularity_baseline_prediction.ipynb`**: 
  - Popularity regression experiments
  - Feature set analysis
  - Regression metrics evaluation
  - Model performance comparison

- **`mult_task_learning.ipynb`**: 
  - Deep learning multi-task architecture
  - Joint training for genre and popularity
  - TensorFlow/Keras implementation
  - Combined loss function optimization

## Configuration

Key parameters that can be adjusted:

- **Train/Val/Test Split**: Currently 70/15/15 (configurable in `data_precessing/datastandardization.py`)
- **Random State**: Set to 42 for reproducibility
- **PCA Variance**: 95% variance retention (configurable in PCA scripts)
- **Model Hyperparameters**: Adjustable in respective notebooks

## Feature Sets

The project experiments with different feature combinations:

- **All Features**: All 12 audio and musical features
- **Core Audio**: `acousticness`, `danceability`, `energy`, `valence`, `tempo`
- **Rhythm**: `danceability`, `tempo`, `energy`, `speechiness`

## Evaluation Metrics

### Classification Metrics
- Accuracy
- Precision (macro-averaged)
- Recall (macro-averaged)
- F1-Score (macro-averaged)
- Confusion Matrix

### Regression Metrics
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Spearman Correlation

## Key Insights

1. **Tree-based models outperform linear models**: Random Forest and XGBoost consistently outperform linear models (Logistic Regression, Ridge, Lasso) for both genre classification and popularity prediction, indicating strong nonlinear relationships between musical characteristics and listener behavior
2. **XGBoost** performs best for genre classification, achieving ~55% accuracy
3. **Random Forest** and **XGBoost** are top performers for popularity prediction
4. Using all features generally outperforms feature subsets
5. Multi-task learning allows shared representation learning between related tasks, enabling the model to leverage common patterns between genre and popularity
6. Genre-based imputation improves handling of missing values by utilizing domain knowledge

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source and available for educational and research purposes.

## Acknowledgments

- Dataset: Music features dataset with genre and popularity labels
- Libraries: scikit-learn, XGBoost, TensorFlow, pandas, numpy

---

**Note**: Make sure to run the scripts in the correct order as they have dependencies on previous outputs. The pipeline is designed to be executed sequentially from `data_precessing/datacleaning.py` through the model training notebooks in `model_training_benchmarking/`. All scripts should be run from their respective directories or with proper path adjustments.
