# ==========================
# STEP 11: Import Libraries
# ==========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# ==========================
# STEP 12: Load Dataset
# ==========================
df = pd.read_csv("data/telco_customer_churn.csv")

# ==========================
# Display First 5 Rows
# ==========================
print("First 5 Rows:")
print(df.head())

# ==========================
# STEP 13: Dataset Shape
# ==========================
print("\nDataset Shape:")
print(df.shape)

# ==========================
# Dataset Information
# ==========================
print("\nDataset Information:")
print(df.info())

# ==========================
# Missing Values
# ==========================
print("\nMissing Values:")
print(df.isnull().sum())

# ==========================
# STEP 14: Convert TotalCharges
# ==========================
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# ==========================
# Check Data Types Again
# ==========================
print("\nUpdated Data Types:")
print(df.dtypes)

# ==========================
# Check Missing Values Again
# ==========================
print("\nMissing Values After Conversion:")
print(df.isnull().sum())

# ==========================
# STEP 15: Fill Missing Values
# ==========================
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Check Missing Values Again
print("\nMissing Values After Filling:")
print(df.isnull().sum())

# ==========================
# STEP 18: Churn Count Plot
# ==========================
plt.figure(figsize=(6,4))

sns.countplot(x="Churn", data=df)

plt.title("Customer Churn Count")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.savefig("images/churn_count.png")

plt.show()

# ==========================
# STEP 19: Gender vs Churn
# ==========================
plt.figure(figsize=(6,4))

sns.countplot(x="gender", hue="Churn", data=df)

plt.title("Gender vs Churn")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")

plt.savefig("images/gender_churn.png")

plt.show()

# ==========================
# STEP 20: Contract vs Churn
# ==========================
plt.figure(figsize=(8,5))

sns.countplot(x="Contract", hue="Churn", data=df)

plt.title("Contract Type vs Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.xticks(rotation=10)

plt.savefig("images/contract_churn.png")

plt.show()

# ==========================
# STEP 21: Internet Service vs Churn
# ==========================
plt.figure(figsize=(7,5))

sns.countplot(x="InternetService", hue="Churn", data=df)

plt.title("Internet Service vs Churn")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.savefig("images/internet_churn.png")

plt.show()

# ==========================
# STEP 22: Monthly Charges Distribution
# ==========================
plt.figure(figsize=(8,5))

sns.histplot(df["MonthlyCharges"], bins=30)

plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Count")

plt.savefig("images/monthly_charges.png")

plt.show()

# ==========================
# STEP 23: Tenure Distribution
# ==========================
plt.figure(figsize=(8,5))

sns.histplot(df["tenure"], bins=30)

plt.title("Customer Tenure Distribution")
plt.xlabel("Tenure (Months)")
plt.ylabel("Count")

plt.savefig("images/tenure_distribution.png")

plt.show()

# ==========================
# STEP 24: Drop CustomerID
# ==========================
df.drop("customerID", axis=1, inplace=True)

# ==========================
# STEP 25: Convert Churn
# ==========================
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# ==========================
# STEP 26: Label Encoding
# ==========================
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for column in df.columns:
    if df[column].dtype == "object" or str(df[column].dtype) == "string":
        df[column] = le.fit_transform(df[column])

# Convert remaining string columns
for column in df.columns:
    if df[column].dtype == "str":
        df[column] = le.fit_transform(df[column])

print("\nData Types After Encoding:")
print(df.dtypes)

# ==========================
# STEP 27: Features and Target
# ==========================
X = df.drop("Churn", axis=1)
y = df["Churn"]
print(X.head())
# ==========================
# STEP 28: Train-Test Split
# ==========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# STEP 29: Train Model
# ==========================
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=5000)

print("\nX Data Types:")
print(X.dtypes)

print("\nColumns with object data:")
print(X.select_dtypes(include=["object"]).columns)

model.fit(X_train, y_train)

# ==========================
# STEP 30: Predictions
# ==========================
y_pred = model.predict(X_test)

# ==========================
# STEP 31: Accuracy
# ==========================
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

# ==========================
# STEP 32: Confusion Matrix Plot
# ==========================
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("images/confusion_matrix.png")

plt.show()

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

importance = rf.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance.head(10))

plt.figure(figsize=(8,6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance.head(10)
)

plt.title("Top 10 Important Features")

plt.savefig("images/feature_importance.png")

plt.show()