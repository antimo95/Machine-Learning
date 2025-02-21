# 🧠 Predicting Multiple Sclerosis Disease Duration from MRI Scans

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange) ![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-red)

## 📌 Project Overview
This project aims to predict the disease duration in patients with Multiple Sclerosis (MS) using MRI scans. The goal is to develop a regression model that minimizes the **Mean Absolute Error (MAE)**, helping clinicians better evaluate MS severity.

## 📂 Dataset
- **Source**: Kaggle Competition
- **Data**: training.csv (labeled)| train.csv
- **Contents**: The dataset consists of MRI scans accompanied by disease duration labels, used to train and evaluate predictive models

## 🛠 Tools & Technologies Used
- **Google Colab** for execution
- **Kaggle API** for data retrieval
- **Python Libraries**:
  - `numpy`, `pandas` for data handling
  - `matplotlib`, `seaborn` for visualization
  - `scikit-learn` for ML models

## 📊 Data Analysis & Preprocessing
### 🔍 Exploratory Data Analysis (EDA)
- Visual analysis of feature distributions
- **Shapiro-Wilk test** for normality check
- Variance analysis to identify impactful features
- **Outlier detection** using **IQR/Box-plot**
- Handling missing values
- **Correlation Analysis**:
  - Evaluated correlation between features and target
  - High correlation among features led to multicollinearity concerns
  - Used **Variance Inflation Factor (VIF)** to confirm high multicollinearity

### 🏗 Dataset Splitting
- **Holdout method** used for train-test split

### 🔽 Dimensionality Reduction
- **Principal Component Analysis (PCA)**
- **Partial Least Squares (PLS)** (chosen for best performance based on MAE)

## 🤖 Model Training & Evaluation
We tested **five regression models**:
- ✅ **Linear Regression**
- ✅ **Random Forest Regressor**
- ✅ **AdaBoost Regressor**
- ✅ **K-Neighbors Regressor**
- ✅ **Support Vector Regressor (SVR)**

🔹 **PLS gave the best results (lowest MAE), so it was selected as the final approach.**

- **Hyperparameter tuning** performed with `RandomizedSearchCV`.
- **Stacking techniques** applied to improve performance.

## 📈 Results
- **Final Model**: PLS + Stacking
- **Evaluation Metric**: Mean Absolute Error (MAE)

📌 Conclusion

Throughout this project, various Machine Learning models were tested for prediction, 
focusing on performance and generalization. Among the approaches analyzed, the most promising candidates were:

Random Forest
Support Vector Regression (SVR) with RBF kernel
Stacking with a linear regression meta-model
Based on the results, the Stacking model with a linear regression meta-model demonstrated the best performance, showing a superior ability to capture relationships in the data and generalize more effectively compared to other approaches.

🙌 Acknowledgments

University of Naples "Federico II" Machine Learning Final Contest
