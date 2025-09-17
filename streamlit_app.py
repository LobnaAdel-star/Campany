import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier   # ✅ LightGBM
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("bank.csv")
    return data

data = load_data()

st.title("Bank Marketing Prediction App")

st.subheader("Dataset Preview")
st.write(data.head())

# Encode categorical variables
data_encoded = data.copy()
for col in data_encoded.select_dtypes(include='object').columns:
    if col != 'deposit':
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col])

X = data_encoded.drop("deposit", axis=1)
y = data_encoded["deposit"].apply(lambda v: 1 if str(v).lower() == "yes" else 0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model selection
st.sidebar.subheader("Choose Model")
model_choice = st.sidebar.selectbox("Select Model", [
    "Logistic Regression",
    "KNN",
    "SVM",
    "Decision Tree",
    "Random Forest",
    "AdaBoost",
    "Gradient Boosting",
    "XGBoost",
    "LightGBM"   # ✅
])

def get_model(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)
    elif name == "KNN":
        return KNeighborsClassifier()
    elif name == "SVM":
        return SVC(probability=True)
    elif name == "Decision Tree":
        return DecisionTreeClassifier()
    elif name == "Random Forest":
        return RandomForestClassifier()
    elif name == "AdaBoost":
        return AdaBoostClassifier()
    elif name == "Gradient Boosting":
        return GradientBoostingClassifier()
    elif name == "XGBoost":
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    elif name == "LightGBM":
        return LGBMClassifier()

model = get_model(model_choice)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("Model Performance")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.write(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
st.write(f"Precision: {precision_score(y_test, y_pred):.2f}")
st.write(f"Recall: {recall_score(y_test, y_pred):.2f}")

# Prediction form
# Prediction form
st.subheader("Try Prediction")
with st.form("prediction_form"):
    inputs = {}
    for col in X.columns:
        if col == "pdays":   # ✅ استبعاد العمود
            continue
        if data[col].dtype == "object":
            options = data[col].unique().tolist()
            val = st.selectbox(f"{col}", options)
            inputs[col] = val
        else:
            val = st.number_input(f"{col}", value=0)
            inputs[col] = val

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([inputs])
        # Encode categorical inputs
        for col in input_df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            input_df[col] = le.fit_transform(input_df[col])
        y_new_pred = model.predict(input_df)
        result = "Yes" if y_new_pred[0] == 1 else "No"
        st.success(f"Prediction: {result}")
