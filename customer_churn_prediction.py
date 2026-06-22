import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
np.random.seed(42)

n = 200

age = np.random.randint(18, 65, n)
monthly_charges = np.random.randint(300, 3000, n)
tenure = np.random.randint(1, 72, n)

contract = np.random.choice(
    ["Month-to-month", "One Year", "Two Year"],
    n
)

internet_service = np.random.choice(
    ["DSL", "Fiber Optic", "No"],
    n
)

# Churn Logic
churn = []

for i in range(n):
    if monthly_charges[i] > 1800 and tenure[i] < 12:
        churn.append("Yes")
    else:
        churn.append("No")

df = pd.DataFrame({
    "Age": age,
    "MonthlyCharges": monthly_charges,
    "Tenure": tenure,
    "Contract": contract,
    "InternetService": internet_service,
    "Churn": churn
})

print("First 5 Rows")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())



le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

print("\nEncoded Data")
print(df.head())



plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["MonthlyCharges"], bins=20, kde=True)
plt.title("Monthly Charges Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["Tenure"], bins=20, kde=True)
plt.title("Tenure Distribution")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\n========== DECISION TREE ==========")

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)

print("Accuracy:", round(dt_acc*100,2), "%")

print("\nClassification Report")
print(classification_report(y_test, dt_pred))

dt_cm = confusion_matrix(y_test, dt_pred)

plt.figure(figsize=(5,4))
sns.heatmap(dt_cm,
            annot=True,
            fmt='d',
            cmap='Blues')

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



print("\n========== RANDOM FOREST ==========")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

print("Accuracy:", round(rf_acc*100,2), "%")

print("\nClassification Report")
print(classification_report(y_test, rf_pred))

rf_cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(5,4))
sns.heatmap(rf_cm,
            annot=True,
            fmt='d',
            cmap='Greens')

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

comparison = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy": [dt_acc, rf_acc]
})

print("\nModel Comparison")
print(comparison)

plt.figure(figsize=(6,4))
sns.barplot(
    x="Model",
    y="Accuracy",
    data=comparison
)

plt.title("Model Accuracy Comparison")
plt.show()


if rf_acc > dt_acc:
    best = "Random Forest"
else:
    best = "Decision Tree"

print("\n========================")
print("BEST MODEL :", best)
print("========================")

print("\nDecision Tree Accuracy :", round(dt_acc*100,2), "%")
print("Random Forest Accuracy :", round(rf_acc*100,2), "%")

