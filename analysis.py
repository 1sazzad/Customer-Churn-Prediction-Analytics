import pandas as pd

# ==============================
# 1. LOAD DATASET
# ==============================

df = pd.read_csv("data/Telco-Customer-Churn.csv")

print("CUSTOMER CHURN ANALYSIS")
print("-" * 40)

print("\nOriginal Shape:", df.shape)


# ==============================
# 2. CLEAN DATA
# ==============================

# Convert TotalCharges to numeric
# Blank values will become NaN
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nMissing Values After Conversion:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Remove rows containing missing TotalCharges
df = df.dropna()

# customerID is only an identifier,
# so we don't need it for machine learning
df = df.drop(columns=["customerID"])


# ==============================
# 3. CHECK CLEAN DATA
# ==============================

print("\nClean Dataset Shape:")
print(df.shape)

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(
    (df["Churn"].value_counts(normalize=True) * 100).round(2)
)