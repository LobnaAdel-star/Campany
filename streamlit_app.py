# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore")

# ================================
# Streamlit UI
# ================================
st.title("📊 Bank Marketing ML Dashboard")
st.markdown("### Compare Machine Learning Models on Bank Marketing Dataset")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("✅ Data Shape:", data.shape)
    st.dataframe(data.head())

    # Missing values
    st.subheader("🔍 Missing Values")
    st.write(data.isnull().sum())

    # Target distribution
    st.subheader("📈 Target Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="deposit", data=data, ax=ax)
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("📊 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # ================================
    # Preprocessing
    # ================================
    X = data.drop("deposit", axis=1)
    y = data["deposit"]

    # Encode target
    y = LabelEncoder().fit_transform(y)

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Models
    models = [
        (LogisticRegression(max_iter=1000), "Logistic Regression"),
        (KNeighborsClassifier(), "KNN"),
        (SVC(), "SVM"),
        (DecisionTreeClassifier(), "Decision Tree"),
        (RandomForestClassifier(), "Random Forest"),
        (AdaBoostClassifier(), "AdaBoost"),
        (GradientBoostingClassifier(), "GradientBoosting"),
        (XGBClassifier(use_label_encoder=False, eval_metric="logloss"), "XGBoost"),
        (LGBMClassifier(), "LightGBM"),
        (CatBoostClassifier(verbose=0), "CatBoost"),
        (GaussianNB(), "Naive Bayes"),
        (MLPClassifier(max_iter=500), "Neural Network")
    ]

    def train_and_evaluate(model, name):
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        return {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred)
        }

    # Run models
    st.subheader("🏆 Models Comparison")
    results = []
    for model, name in models:
        results.append(train_and_evaluate(model, name))

    results_df = pd.DataFrame(results)
    st.dataframe(results_df.sort_values(by="F1", ascending=False))

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 5))
    results_df.set_index("Model")["F1"].plot(kind="bar", ax=ax, title="Model F1-Score Comparison")
    st.pyplot(fig)

    # ================================
    # Hyperparameter Tuning
    # ================================
    st.subheader("🔧 Hyperparameter Tuning (Random Forest)")

    best_model = RandomForestClassifier()
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=Pipeline(steps=[("preprocess", preprocessor), ("model", best_model)]),
        param_grid={"model__" + k: v for k, v in param_grid.items()},
        cv=3,
        scoring="f1",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    st.write("Best Parameters:", grid_search.best_params_)

    best_pipe = grid_search.best_estimator_
    y_pred_best = best_pipe.predict(X_test)

    st.write("Accuracy:", accuracy_score(y_test, y_pred_best))
    st.write("Precision:", precision_score(y_test, y_pred_best))
    st.write("Recall:", recall_score(y_test, y_pred_best))
    st.write("F1:", f1_score(y_test, y_pred_best))
else:
    st.info("👆 Please upload your `bank.csv` file to start.")
