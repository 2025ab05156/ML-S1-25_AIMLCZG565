import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="ML Classification Models - Employee Burnout",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Title and description
st.title("🤖 Work From Home Employee Burnout Classification")
st.markdown("""
    ### Interactive Machine Learning Classification System
    Demonstrating 6 different classification models on the Work From Home Employee Burnout Dataset
    """)
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Select Mode:",
    ["📊 Dataset Overview", "📈 Model Comparison", "🎯 Make Predictions", "ℹ️ About Models"]
)

@st.cache_data
def load_and_prepare_data():
    """Load Work From Home Employee Burnout dataset from Kaggle"""
    try:
        import kagglehub
        import glob
        
        # Load dataset and get the path
        dataset_path = kagglehub.load_dataset(
            "sonalshinde123/work-from-home-employee-burnout-dataset"
        )
        
        # Find CSV files in the dataset path
        csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
        
        if csv_files:
            df = pd.read_csv(csv_files[0])
            return df
        else:
            raise Exception("No CSV files found in the dataset")
            
    except Exception as e:
        # Fallback: Try loading from local cache
        if os.path.exists("burnout_data.csv"):
            return pd.read_csv("burnout_data.csv")
        
        # Fallback: Create sample dataset for demonstration
        np.random.seed(42)
        n_samples = 500
        
        df = pd.DataFrame({
            'Employee_ID': range(1, n_samples + 1),
            'Company_Type': np.random.choice(['Service', 'Product'], n_samples),
            'WFH_Setup_Year': np.random.randint(2015, 2023, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Designation': np.random.choice(['Executive', 'Manager', 'Developer'], n_samples),
            'Resource_Allocation': np.random.uniform(0.5, 1.0, n_samples),
            'Mental_Fatigue_Score': np.random.randint(0, 10, n_samples),
            'Burn_Rate': np.random.choice([0, 1], n_samples)
        })
        
        st.info("📌 Using sample dataset for demonstration (Kaggle API unavailable)")
        return df

@st.cache_data
def load_results():
    """Load pre-computed model results"""
    try:
        results_df = pd.read_csv('model/evaluation_results.csv', index_col=0)
        return results_df
    except:
        return None

@st.cache_data
def load_models():
    """Load trained models from disk"""
    models = {}
    model_files = [
        'logistic_regression',
        'decision_tree',
        'knn',
        'naive_bayes',
        'random_forest',
        'xgboost'
    ]
    
    for model_name in model_files:
        try:
            model_path = f'model/{model_name}_model.pkl'
            if os.path.exists(model_path):
                models[model_name.replace('_', ' ').title()] = joblib.load(model_path)
        except Exception as e:
            st.warning(f"Could not load {model_name}: {e}")
    
    try:
        scaler = joblib.load('model/scaler.pkl')
        return models, scaler
    except:
        return models, None

# Main content
if app_mode == "📊 Dataset Overview":
    st.header("Dataset Overview")
    
    df = load_and_prepare_data()
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Number of Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Data Shape", f"{df.shape[0]} × {df.shape[1]}")
        
        st.subheader("Dataset Statistics")
        st.dataframe(df.describe().T)
        
        st.subheader("First Few Rows")
        st.dataframe(df.head(10))
        
        st.subheader("Data Types")
        st.dataframe(df.dtypes)
        
        st.subheader("Missing Values")
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum(),
            'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(missing_df[missing_df['Missing Count'] > 0])
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Numerical Features Distribution")
            numerical_cols = df.select_dtypes(include=[np.number]).columns[:3]
            for col in numerical_cols:
                fig, ax = plt.subplots()
                ax.hist(df[col], bins=30, color='#4ecdc4', edgecolor='black')
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
        
        with col2:
            st.subheader("Categorical Features")
            categorical_cols = df.select_dtypes(include=['object']).columns[:3]
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    fig, ax = plt.subplots()
                    df[col].value_counts().plot(kind='bar', ax=ax, color='#f7b731')
                    ax.set_title(f'Distribution of {col}')
                    st.pyplot(fig)
    else:
        st.error("Could not load dataset. Please ensure you have kagglehub installed.")

elif app_mode == "📈 Model Comparison":
    st.header("Model Performance Comparison")
    
    results_df = load_results()
    
    if results_df is not None:
        st.subheader("Performance Metrics Summary")
        st.dataframe(results_df.round(4))
        
        # Create columns for visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Accuracy Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            results_df['Accuracy'].sort_values(ascending=True).plot(kind='barh', ax=ax, color='#4ecdc4')
            ax.set_xlabel('Accuracy Score')
            ax.set_title('Model Accuracy Comparison')
            for i, v in enumerate(results_df['Accuracy'].sort_values(ascending=True)):
                ax.text(v + 0.01, i, f'{v:.4f}', va='center')
            st.pyplot(fig)
        
        with col2:
            st.subheader("AUC Score Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            results_df['AUC Score'].sort_values(ascending=True).plot(kind='barh', ax=ax, color='#f7b731')
            ax.set_xlabel('AUC Score')
            ax.set_title('Model AUC Comparison')
            for i, v in enumerate(results_df['AUC Score'].sort_values(ascending=True)):
                ax.text(v + 0.01, i, f'{v:.4f}', va='center')
            st.pyplot(fig)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("F1 Score Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            results_df['F1 Score'].sort_values(ascending=True).plot(kind='barh', ax=ax, color='#a29bfe')
            ax.set_xlabel('F1 Score')
            ax.set_title('Model F1 Score Comparison')
            for i, v in enumerate(results_df['F1 Score'].sort_values(ascending=True)):
                ax.text(v + 0.01, i, f'{v:.4f}', va='center')
            st.pyplot(fig)
        
        with col4:
            st.subheader("All Metrics Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(results_df.round(4), annot=True, fmt='.4f', cmap='YlGnBu', ax=ax, 
                       cbar_kws={'label': 'Score'})
            ax.set_title('Complete Metrics Heatmap')
            st.pyplot(fig)
        
        # Model Ranking
        st.subheader("Model Ranking by Accuracy")
        ranked = results_df.sort_values('Accuracy', ascending=False)
        ranked_display = ranked.copy()
        ranked_display['Rank'] = range(1, len(ranked) + 1)
        ranked_display = ranked_display[['Rank', 'Accuracy', 'AUC Score', 'F1 Score', 'Precision', 'Recall', 'MCC Score']]
        st.dataframe(ranked_display.round(4))
        
        # Best model insights
        best_model = results_df['Accuracy'].idxmax()
        best_accuracy = results_df.loc[best_model, 'Accuracy']
        st.success(f"🏆 Best Model: **{best_model}** with Accuracy: **{best_accuracy:.4f}**")
    else:
        st.warning("Model results not found. Please run train_models.py first.")
        st.info("""
        To generate model results:
        1. Install dependencies: `pip install -r requirements.txt`
        2. Run training script: `python train_models.py`
        3. Refresh this page
        """)

elif app_mode == "🎯 Make Predictions":
    st.header("Make Predictions on New Data")
    
    models, scaler = load_models()
    
    if len(models) > 0 and scaler is not None:
        df = load_and_prepare_data()
        
        if df is not None:
            st.subheader("Input Features for Prediction")
            st.info("Adjust the feature values below to make predictions with all trained models")
            
            # Create input fields for numerical features only
            numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target column if exists
            if 'Burn_Rate' in numerical_features:
                numerical_features.remove('Burn_Rate')
            
            feature_inputs = {}
            
            # Create dynamic input fields
            cols = st.columns(3)
            for idx, feature in enumerate(numerical_features):
                col_idx = idx % 3
                with cols[col_idx]:
                    feature_inputs[feature] = st.number_input(
                        f"{feature}",
                        value=float(df[feature].mean()),
                        min_value=float(df[feature].min()),
                        max_value=float(df[feature].max()),
                        step=0.1
                    )
            
            if st.button("🔮 Make Prediction", key="predict_button"):
                # Prepare input
                input_data = pd.DataFrame([feature_inputs])
                input_scaled = scaler.transform(input_data)
                
                st.subheader("Prediction Results from All Models")
                
                prediction_results = []
                
                cols = st.columns(3)
                for idx, (name, model) in enumerate(models.items()):
                    try:
                        prediction = model.predict(input_scaled)[0]
                        probability = model.predict_proba(input_scaled)[0]
                        
                        pred_label = "🟢 No Burnout (Class 1)" if prediction == 1 else "🔴 Burnout (Class 0)"
                        confidence = max(probability) * 100
                        
                        prediction_results.append({
                            'Model': name,
                            'Prediction': pred_label,
                            'Confidence': f"{confidence:.2f}%"
                        })
                        
                        col_idx = idx % 3
                        with cols[col_idx]:
                            st.metric(name, pred_label, f"Confidence: {confidence:.1f}%")
                    except Exception as e:
                        st.warning(f"Error with {name}: {e}")
                
                # Display summary table
                if prediction_results:
                    st.subheader("Summary Table")
                    summary_df = pd.DataFrame(prediction_results)
                    st.dataframe(summary_df)
    else:
        st.error("Models not found!")
        st.warning("""
        **To use predictions:**
        1. Run the training script: `python train_models.py`
        2. This will train and save all 6 models
        3. Refresh this page
        """)

elif app_mode == "ℹ️ About Models":
    st.header("About the Classification Models")
    
    models_info = {
        "Logistic Regression": {
            "Description": "A linear model for binary classification that uses a sigmoid function to map inputs to probabilities.",
            "Pros": "Simple, interpretable, fast, works well for linearly separable data",
            "Cons": "Limited to linear relationships",
            "Best for": "Quick baseline models, when interpretability is important"
        },
        "Decision Tree": {
            "Description": "A tree-based model that makes decisions by splitting data based on feature values.",
            "Pros": "Interpretable, handles non-linear relationships, no scaling needed",
            "Cons": "Prone to overfitting, unstable",
            "Best for": "When model interpretability is crucial"
        },
        "K-Nearest Neighbor": {
            "Description": "An instance-based learning algorithm that classifies based on the k nearest neighbors.",
            "Pros": "Simple to understand and implement, no training phase",
            "Cons": "Slow predictions, sensitive to feature scaling, high memory usage",
            "Best for": "Small datasets, when simplicity is preferred"
        },
        "Naive Bayes": {
            "Description": "A probabilistic classifier based on Bayes' theorem with strong independence assumptions.",
            "Pros": "Fast, works well with small datasets, probabilistic interpretations",
            "Cons": "Assumes feature independence (which rarely holds)",
            "Best for": "Text classification, spam detection, when training data is limited"
        },
        "Random Forest": {
            "Description": "An ensemble method that combines multiple decision trees with bootstrap aggregating.",
            "Pros": "Robust, handles non-linear relationships, parallel processing, feature importance",
            "Cons": "More complex, slower predictions, harder to interpret",
            "Best for": "General-purpose classification, when accuracy is priority"
        },
        "XGBoost": {
            "Description": "An advanced gradient boosting ensemble method that sequentially builds trees.",
            "Pros": "Excellent accuracy, handles complex patterns, built-in regularization",
            "Cons": "Computationally expensive, prone to overfitting if not tuned properly",
            "Best for": "Competitions, when highest accuracy is needed"
        }
    }
    
    st.subheader("Evaluation Metrics Explained")
    
    metrics_info = {
        "Accuracy": "Percentage of correct predictions out of total predictions. Overall measure of model performance.",
        "AUC Score": "Area Under the ROC Curve. Measures the model's ability to distinguish between classes (0-1).",
        "Precision": "Ratio of true positives to all positive predictions. Answers: Of predicted positives, how many were correct?",
        "Recall": "Ratio of true positives to all actual positives. Answers: Of actual positives, how many did we find?",
        "F1 Score": "Harmonic mean of Precision and Recall. Balances both metrics.",
        "MCC Score": "Matthews Correlation Coefficient. Balanced metric even for imbalanced datasets (-1 to 1)."
    }
    
    tab1, tab2 = st.tabs(["Models", "Metrics"])
    
    with tab1:
        selected_model = st.selectbox("Select a model to learn more:", list(models_info.keys()))
        
        if selected_model:
            info = models_info[selected_model]
            st.subheader(selected_model)
            st.write(f"**Description:** {info['Description']}")
            st.write(f"**Pros:** {info['Pros']}")
            st.write(f"**Cons:** {info['Cons']}")
            st.write(f"**Best for:** {info['Best for']}")
    
    with tab2:
        selected_metric = st.selectbox("Select a metric to learn more:", list(metrics_info.keys()))
        
        if selected_metric:
            st.subheader(selected_metric)
            st.write(metrics_info[selected_metric])
    
    st.markdown("---")
    st.subheader("Dataset Information")
    st.write("""
    **Work From Home Employee Burnout Dataset**
    - Source: Kaggle (sonalshinde123/work-from-home-employee-burnout-dataset)
    - Target: Burn_Rate (burnout classification)
    - Features: Various employee characteristics and work metrics
    - Use Case: Predicting employee burnout risk for WFH employees
    """)
    
    st.subheader("How to Use This App")
    st.markdown("""
    1. **📊 Dataset Overview**: Explore the dataset structure and statistics
    2. **📈 Model Comparison**: Compare performance of all 6 trained models
    3. **🎯 Make Predictions**: Input feature values to get predictions from all models
    4. **ℹ️ About Models**: Learn about each model and evaluation metrics
    """)

st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | Machine Learning | Data Science</p>
    </div>
""", unsafe_allow_html=True)
