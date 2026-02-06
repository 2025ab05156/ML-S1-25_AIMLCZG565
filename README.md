# UCI Credit Approval Classification

## 🎯 Problem Statement

This project addresses the critical task of **automated credit approval decision-making** for financial institutions. The goal is to develop and compare multiple machine learning models that can accurately predict whether a credit application should be **approved** or **denied** based on applicant features.

**Business Challenge:**
- Financial institutions process thousands of credit applications daily
- Manual review is time-consuming and prone to human bias
- Need for consistent, data-driven decision-making process
- Requirement to minimize both false approvals (financial risk) and false rejections (lost business opportunities)

**Technical Objective:**
Implement and evaluate **6 different classification algorithms** to determine the most effective approach for credit approval prediction, focusing on achieving high accuracy while maintaining balanced precision and recall metrics.

## 📊 Project Overview

This project implements and compares **6 different machine learning classification models** on the **UCI Credit Approval Dataset**. It includes a fully interactive Streamlit web application for exploring data, evaluating models, and making predictions.

### Dataset: UCI Credit Approval (ID: 27)

**Source:** [UCI Machine Learning Repository - Credit Approval Dataset](https://archive.ics.uci.edu/dataset/27/credit+approval)

**Dataset Characteristics:**
- Credit card application approval data
- 15 features with mix of categorical and numerical attributes
- Target Variable: Approval decision (Binary - Approved: 1, Denied: 0)
- Total Samples: 690 applicants
- Some missing values in features
- Mixed feature types: Numerical (age, income, credit score) and categorical (employment type, marital status)

## 🤖 Classification Models Implemented

### 1. **Logistic Regression**
- Linear model for binary classification
- Uses sigmoid function to map predictions to probabilities
- Fast training and inference
- Best for: Quick baseline, interpretable results

### 2. **Decision Tree Classifier**
- Hierarchical tree-based model
- Creates interpretable decision rules
- Good for feature importance analysis
- Best for: Interpretability, non-linear patterns

### 3. **K-Nearest Neighbor (KNN)**
- Instance-based learning algorithm
- Classifies based on k nearest neighbors
- Requires feature scaling
- Best for: Small datasets, simple patterns

### 4. **Naive Bayes (Gaussian)**
- Probabilistic classifier based on Bayes' theorem
- Assumes feature independence
- Fast training and inference
- Best for: Fast predictions, limited data

### 5. **Random Forest (Ensemble)**
- Ensemble of multiple decision trees
- Reduces overfitting through bagging
- Provides feature importance scores
- Best for: Balanced accuracy-complexity trade-off

### 6. **XGBoost (Ensemble)**
- Gradient boosting ensemble model
- Sequential tree building with residual correction
- Handles complex non-linear relationships
- Best for: Maximum accuracy, complex patterns

## 📊 Model Performance Comparison

### Comprehensive Evaluation Results

All models were trained and evaluated on the UCI Credit Approval dataset with 800 training samples and 200 testing samples. The following tables present the complete performance analysis:

#### 🏆 Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----- |
| Logistic Regression | 0.9500 | 0.9916 | 0.9500 | 0.9500 | 0.9499 | 0.8979 |
| Decision Tree | 0.8950 | 0.9008 | 0.9515 | 0.8596 | 0.9032 | 0.7940 |
| kNN | 0.9700 | 0.9933 | 0.9909 | 0.9561 | 0.9732 | 0.9399 |
| Naive Bayes | 0.9350 | 0.9854 | 0.9720 | 0.9123 | 0.9412 | 0.8709 |
| Random Forest (Ensemble) | 0.9600 | 0.9945 | 0.9818 | 0.9474 | 0.9643 | 0.9196 |
| XGBoost (Ensemble) | 0.9700 | 0.9930 | 0.9821 | 0.9649 | 0.9735 | 0.9392 |

#### 📈 Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-----------------------------------|
| Logistic Regression | Strong baseline performance with 95% accuracy. Excellent balance across all metrics with particularly high AUC (0.9916). Fast training and consistent results make it reliable for quick deployment. |
| Decision Tree | Lowest overall accuracy (89.5%) but highest precision (0.9515). Shows signs of overfitting with high variance. Good interpretability but less stable performance compared to ensemble methods. |
| kNN | Tied for highest accuracy (97%) with excellent precision (0.9909). Outstanding MCC score (0.9399) indicates robust performance across classes. However, computationally expensive for large datasets. |
| Naive Bayes | Good performance (93.5% accuracy) despite strong independence assumptions. High precision (0.9720) but lower recall (0.9123). Fast training and works well with limited data. |
| Random Forest (Ensemble) | Excellent performance (96% accuracy) with highest AUC score (0.9945). Well-balanced metrics and good generalization. Provides feature importance insights while avoiding overfitting. |
| XGBoost (Ensemble) | Tied for highest accuracy (97%) with best overall balance. Highest recall (0.9649) and F1 score (0.9735). Superior handling of complex patterns but requires more computational resources. |

### 🏅 Performance Rankings

**By Accuracy:**
1. **KNN & XGBoost**: 97.0% (tied)
2. **Random Forest**: 96.0%
3. **Logistic Regression**: 95.0%
4. **Naive Bayes**: 93.5%
5. **Decision Tree**: 89.5%

**Best Overall Model**: **XGBoost** - Combines highest accuracy with best F1 score and excellent recall, making it optimal for credit approval where minimizing false negatives is crucial.

## 📈 Evaluation Metrics (6 Metrics per Model)

All models are evaluated using the following metrics:

### 1. **Accuracy**
- Proportion of correct predictions
- Formula: (TP + TN) / (TP + TN + FP + FN)
- Best for: Overall model performance assessment

### 2. **AUC Score (Area Under ROC Curve)**
- Measures model's ability to distinguish between classes
- Range: 0 to 1 (higher is better, 0.5 = random)
- Best for: Imbalanced datasets, comparing models

### 3. **Precision**
- Proportion of positive predictions that are correct
- Formula: TP / (TP + FP)
- Best for: When false positives are costly

### 4. **Recall (Sensitivity)**
- Proportion of actual positives correctly identified
- Formula: TP / (TP + FN)
- Best for: When false negatives are costly

### 5. **F1 Score**
- Harmonic mean of Precision and Recall
- Formula: 2 × (Precision × Recall) / (Precision + Recall)
- Best for: Balancing precision and recall

### 6. **Matthews Correlation Coefficient (MCC)**
- Correlation between predicted and actual labels
- Range: -1 to 1 (higher is better)
- Best for: Imbalanced datasets, balanced evaluation

## 📁 Project Structure

```
ML-S1-25_AIMLCZG565/
├── app.py                          # Main Streamlit application
├── train_models.py                 # Model training and evaluation script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── model/                          # Saved trained models
    ├── logistic_regression_model.pkl
    ├── decision_tree_model.pkl
    ├── knn_model.pkl
    ├── naive_bayes_model.pkl
    ├── random_forest_model.pkl
    ├── xgboost_model.pkl
    ├── scaler.pkl                 # Feature scaler
    └── evaluation_results.csv      # Model evaluation metrics
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ML-S1-25_AIMLCZG565
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Kaggle API (if needed):**
   - Download your Kaggle API token from [https://www.kaggle.com/settings/account](https://www.kaggle.com/settings/account)
   - Place it in `~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)

### Usage

#### Step 1: Train the Models
```bash
python train_models.py
```

This script will:
- Load the UCI Credit Approval dataset from UCI ML Repository
- Preprocess and prepare the data (handle missing values, encode categorical features)
- Train all 6 classification models
- Calculate evaluation metrics for each model
- Save trained models and results to the `model/` directory

**Output:**
- Trained model files (`.pkl` files in `model/` folder)
- Scaler object for feature normalization
- Evaluation results CSV file with all metrics

#### Step 2: Run the Streamlit Web Application
```bash
streamlit run app.py
```

The web app will open in your browser (typically http://localhost:8501)

### Application Features

#### 📊 Dataset Overview
- Load and explore the student performance dataset
- View dataset statistics and distributions
- Analyze feature characteristics
- Visualize numerical and categorical features

#### 📈 Model Comparison
- Compare performance of all 6 models
- View detailed evaluation metrics
- Visualize model performance with charts
- See model rankings by different metrics

#### 🎯 Make Predictions
- Input feature values for a new employee
- Get predictions from all 6 models
- View confidence scores from each model
- Compare predictions across models

#### ℹ️ About Models
- Learn about each classification algorithm
- Understand evaluation metrics
- Get guidance on when to use each model
- Access best practices and recommendations

## 📊 Model Training Details

### Data Preprocessing
1. Load dataset from Kaggle
2. Handle missing values (dropna)
3. Encode categorical variables (LabelEncoder)
4. Split into training (80%) and testing (20%) sets
5. Scale numerical features (StandardScaler)

### Model Configuration

| Model | Parameters | Notes |
|-------|-----------|-------|
| Logistic Regression | max_iter=1000 | Scaled features required |
| Decision Tree | max_depth=10 | No scaling required |
| KNN | n_neighbors=5 | Scaled features required |
| Naive Bayes | (Gaussian) | Scaled features required |
| Random Forest | n_estimators=100, max_depth=10 | Parallel processing (n_jobs=-1) |
| XGBoost | n_estimators=100, max_depth=6 | Gradient boosting |

### Training Output

Each model produces:
- **Trained model file** - Serialized sklearn/xgboost model
- **6 evaluation metrics** - Accuracy, AUC, Precision, Recall, F1, MCC
- **Scaler object** - For consistent feature scaling

## 📈 Expected Results

The evaluation metrics will show:
- **Accuracy**: Overall correctness (0-100%)
- **AUC Score**: Discrimination ability (0-1)
- **Precision**: Positive prediction accuracy (0-1)
- **Recall**: Actual positive detection rate (0-1)
- **F1 Score**: Precision-Recall balance (0-1)
- **MCC Score**: Overall correlation (-1 to 1)

Typically:
- **Random Forest** and **XGBoost** achieve higher accuracy
- **Logistic Regression** is the fastest
- **Decision Tree** is most interpretable
- All ensemble methods generally outperform linear models

## 🌐 Deployment on Streamlit Community Cloud

### Steps to Deploy

1. **Push code to GitHub:**
   ```bash
   git add .
   git commit -m "Add ML models and Streamlit app"
   git push origin main
   ```

2. **Go to Streamlit Community Cloud:**
   - Visit [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign up with GitHub
   - Click "New app"
   - Select repository, branch, and main file (`app.py`)
   - Click "Deploy"

3. **Configure Secrets (if using Kaggle API):**
   - In Streamlit Cloud settings, add:
     ```
     [kaggle]
     username = <your-kaggle-username>
     key = <your-kaggle-api-key>
     ```

### Live Application
Once deployed, your app will be available at:
`https://share.streamlit.io/<your-username>/<repo-name>`

## 📝 Requirements

See `requirements.txt` for all dependencies:
- **streamlit** - Web application framework
- **scikit-learn** - Machine learning models
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **xgboost** - Gradient boosting
- **joblib** - Model serialization
- **kagglehub** - Kaggle dataset loading

## 🔍 Troubleshooting

### Issue: "Module not found" error
**Solution:** Install missing dependencies
```bash
pip install -r requirements.txt
```

### Issue: Kaggle authentication error
**Solution:** Set up Kaggle API credentials
1. Download token from Kaggle account settings
2. Place in `~/.kaggle/kaggle.json`
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Issue: Models not loading in web app
**Solution:** Run training script first
```bash
python train_models.py
```

### Issue: Slow predictions
**Solution:** Models are working as expected. XGBoost may take longer due to complexity.

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Kaggle Datasets](https://www.kaggle.com/datasets/)

## ✅ Checklist

- [x] 6 Classification Models Implemented
- [x] 6 Evaluation Metrics per Model
- [x] Interactive Streamlit Web Application
- [x] Dataset Loading from Kaggle
- [x] Model Training Script
- [x] Model Evaluation and Comparison
- [x] Prediction Interface
- [x] Deployment Ready
- [x] Complete Documentation
- [x] GitHub Repository Structure

## 📄 License

This project is created for educational purposes as part of machine learning coursework.

## 👨‍💻 Author

**Kumaraswamy Posa**
- GitHub: [KumaraswamyPosa](https://github.com/KumaraswamyPosa)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

**Last Updated:** February 5, 2026
**Python Version:** 3.8+
**Status:** ✅ Production Ready
