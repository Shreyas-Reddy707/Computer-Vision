# 🏡 Real Estate Valuation & Price Prediction Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shreyas-Reddy707/Machine-Learning/blob/main/Machine_Learning.ipynb)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)

## 📌 Project Overview
The **Real Estate Valuation & Price Prediction Engine** is an end-to-end Machine Learning regression pipeline designed to estimate residential property prices in INR (₹) based on structural, physical, and locational attributes. 

Developed as an academic and portfolio showcase for **Machine Learning**, this project demonstrates the complete lifecycle of a data science workflow—from synthetic data modeling and exploratory feature importance analysis to hyperparameter tuning (`GridSearchCV`), custom feature engineering, and production-style **Scikit-Learn Pipelines** validated via K-Fold Cross-Validation.

---

## 🎯 Why We Built This Project
Real estate valuation is inherently complex because property prices are driven by multiple interacting factors—square footage, room density, neighborhood quality, and age-related depreciation. Relying on simple rule-of-thumb estimates or basic linear models often fails to capture non-linear market dynamics.

We built this project to:
1. **Simulate Realistic Market Dynamics:** Generate a controlled, noise-injected dataset of 500 properties where positive value drivers (area, location rating, nearby amenities) and negative drivers (property age/depreciation) interact.
2. **Benchmark Single Trees vs. Ensembles:** Prove empirically why ensemble methods (**Random Forest** and **Gradient Boosting**) significantly outperform standalone **Decision Trees** in reducing variance and prediction error.
3. **Demonstrate Leak-Free ML Engineering:** Show how to properly bundle feature scaling (`StandardScaler`) and regression estimators inside unified `Pipeline` objects evaluated with 5-Fold Cross-Validation.

---

## 📊 Dataset & Feature Specifications
The synthetic dataset consists of **500 residential records** (`np.random.seed(42)`) with zero missing values across 6 core input features and 2 engineered features:

### Core Features
| Feature | Type | Range / Scale | Description |
| :--- | :--- | :--- | :--- |
| `Area` | `int64` | 500 – 4,000 sq. ft. | Total built-up area of the house (primary price driver). |
| `Bedrooms` | `int64` | 1 – 5 units | Number of bedrooms in the property. |
| `Bathrooms` | `int64` | 1 – 4 units | Number of bathroom units. |
| `Location_Rating` | `int64` | 1 – 10 scale | Subjective neighborhood premium & connectivity score. |
| `Age` | `int64` | 1 – 30 years | Age of the property (negatively impacts valuation via depreciation). |
| `Nearby_Amenities` | `int64` | 1 – 15 count | Number of essential facilities (schools, hospitals, transit) nearby. |
| **`Price` (Target)** | `int64` | ₹26.1L – ₹1.75Cr | Estimated market valuation of the home in INR (₹). |

### Engineered Features
* **`Bed_Bath_Ratio`**: $\frac{\text{Bedrooms}}{\text{Bathrooms} + 0.1}$ — Captures layout balance while preventing division-by-zero errors.
* **`Area_per_Room`**: $\frac{\text{Area}}{\text{Bedrooms} + \text{Bathrooms} + 1}$ — Measures room spaciousness and density.

---

## ⚙️ Project Architecture & Workflow

1. **Synthetic Data Generation & Inspection:** Synthesizes 500 housing records with realistic feature weights and random market noise ($\pm ₹5,00,000$).
2. **Train-Test Partitioning:** Splits data into **80% Training (400 samples)** and **20% Testing (100 samples)** sets.
3. **Baseline Modeling & Feature Importance:** Fits a constrained `DecisionTreeRegressor(max_depth=7)` and extracts feature importance scores, confirming `Area` as the dominant predictor followed by `Age` and `Location_Rating`.
4. **Out-of-Sample Inference:** Tests real-time prediction capability on unseen manual property profiles (e.g., a 2,400 sq. ft., 3-BHK home predicts at **~₹1.10 Cr**).
5. **Ensemble Upgrades & Hyperparameter Tuning:** Uses 5-fold `GridSearchCV` across `n_estimators`, `max_depth`, and `min_samples_split` to optimize a `RandomForestRegressor` (`max_depth: 15`, `min_samples_split: 2`, `n_estimators: 200`) and compares it against `GradientBoostingRegressor`.
6. **Scikit-Learn Pipelines & Cross-Validation:** Wraps `StandardScaler` and each regressor into modular `Pipeline` objects, evaluating stability across 5-Fold Cross-Validation.

---

## 📈 Model Performance & Results

### 1. Standard Model Comparison (Test Set)
| Model | MAE (₹) | RMSE (₹) | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **Gradient Boosting (Base)** | **₹3,20,323** | **₹4,23,032** | **0.9858** |
| **Random Forest (GridSearchCV Tuned)** | ₹4,20,673 | ₹5,40,195 | 0.9768 |
| **Decision Tree (Baseline)** | ₹6,26,687 | ₹7,99,136 | 0.9492 |

### 2. Engineered Features + Scikit-Learn Pipeline (5-Fold CV)
| Pipeline Model | CV Mean $R^2$ | CV Std $R^2$ | Test $R^2$ | Test MAE (₹) |
| :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting (Pipeline)** | **0.9873** | **±0.0008** | **0.9855** | **₹3,22,954** |
| **Random Forest (Pipeline)** | 0.9783 | ±0.0019 | 0.9754 | ₹4,35,969 |
| **Decision Tree (Pipeline)** | 0.9550 | ±0.0032 | 0.9448 | ₹6,75,381 |

### Key Takeaways
* **Gradient Boosting** achieved the highest accuracy, explaining **98.58%** of the variance in housing prices and cutting the Mean Absolute Error (MAE) in half compared to the baseline Decision Tree (from ₹6.26 Lakhs down to ₹3.20 Lakhs).
* **Cross-Validation Stability:** Gradient Boosting maintained an ultra-low standard deviation of `0.0008` across 5 folds, proving high generalization without overfitting.

---

## 🚀 Getting Started

### Option 1: Run in Google Colab (Recommended)
Click the badge below to open and execute the notebook directly in the cloud with zero local setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shreyas-Reddy707/Machine-Learning/blob/main/Machine_Learning.ipynb)

### Option 2: Run Locally
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Shreyas-Reddy707/Machine-Learning.git](https://github.com/Shreyas-Reddy707/Machine-Learning.git)
   cd Machine-Learning
