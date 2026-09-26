import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna()


# ==============================
# 1. CHURN DISTRIBUTION
# ==============================

df["Churn"].value_counts().plot(
    kind="bar"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()


# ==============================
# 2. CHURN BY CONTRACT
# ==============================

pd.crosstab(
    df["Contract"],
    df["Churn"]
).plot(kind="bar")

plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ==============================
# 3. TENURE BY CHURN
# ==============================

df.boxplot(
    column="tenure",
    by="Churn"
)

plt.title("Tenure by Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.tight_layout()
plt.show()


# ==============================
# 4. MONTHLY CHARGES BY CHURN
# ==============================

df.boxplot(
    column="MonthlyCharges",
    by="Churn"
)

plt.title("Monthly Charges by Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.show()