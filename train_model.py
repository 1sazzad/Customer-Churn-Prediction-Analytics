import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/Telco-Customer-Churn.csv")

print("CUSTOMER CHURN MODEL TRAINING")
print("-" * 45)


# ==========================================
# 2. CLEAN DATA
# ==========================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna()

# Customer ID is not useful for prediction
df = df.drop(columns=["customerID"])

# Convert target:
# No = 0
# Yes = 1
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["Churn"])
y = df["Churn"]

print("\nDataset:", X.shape)
print("Churn:", y.value_counts().to_dict())


# ==========================================
# 4. COLUMN TYPES
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical Features:", len(categorical_columns))
print("Numerical Features:", len(numerical_columns))


# ==========================================
# 5. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        )
    ]
)


# ==========================================
# 6. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# ==========================================
# 7. MODELS
# ==========================================

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                random_state=42
            )
        )
    ]
)


# ==========================================
# 8. TRAIN AND EVALUATE
# ==========================================

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}

results = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    }


# ==========================================
# 9. SHOW RESULTS
# ==========================================

results_df = pd.DataFrame(results).T

print("\nMODEL RESULTS")
print("-" * 45)

print(
    results_df.round(4)
)


# ==========================================
# 10. SELECT BEST MODEL
# ==========================================

best_model_name = results_df[
    "F1 Score"
].idxmax()

best_model = models[best_model_name]

print("\nBest Model:", best_model_name)


# ==========================================
# 11. SAVE MODEL
# ==========================================

joblib.dump(
    best_model,
    "churn_model.pkl"
)

print("\nModel saved successfully:")
print("churn_model.pkl")