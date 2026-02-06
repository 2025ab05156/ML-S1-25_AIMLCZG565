# UCI Student Performance Classification

## 📊 Project Overview

This project implements and compares **6 different machine learning classification models** on the **UCI Student Performance Dataset**. It includes a fully interactive Streamlit web application for exploring data, evaluating models, and making predictions.

### Dataset: UCI Student Performance (ID: 320)

**Source:** [UCI Machine Learning Repository - Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

**Dataset Characteristics:**
- Student academic performance data from Portuguese schools
- 33 features including demographic, social, and academic information
- Target Variable: Final grade classification (Multiclass - 20 grades: 0-19)
- Features: School, age, study time, past failures, family size, parent education, etc.
- Total Samples: 649 students
- No missing values
- Mixed feature types: Numerical and categorical

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
- Load the UCI Student Performance dataset from UCI ML Repository
- Preprocess and prepare the data
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
