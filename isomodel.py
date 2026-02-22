import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1️⃣ Load preprocessed data
df = pd.read_csv("synthetic_aml_transactions_preprocessed.csv")

# 2️⃣ Train Isolation Forest WITHOUT forcing anomaly %
iso = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42
)
iso.fit(df)

# 3️⃣ Compute anomaly scores for ALL transactions
df["decision_score"] = iso.decision_function(df)   # higher = more normal
df["anomaly_score"] = -df["decision_score"]        # higher = more suspicious

# 4️⃣ Define Risk Levels using percentiles
low_threshold = df["anomaly_score"].quantile(0.80)   # Top 20% = Medium+
high_threshold = df["anomaly_score"].quantile(0.95)  # Top 5% = High

def assign_risk(score):
    if score >= high_threshold:
        return "High Risk"
    elif score >= low_threshold:
        return "Medium Risk"
    else:
        return "Low Risk"

df["risk_level"] = df["anomaly_score"].apply(assign_risk)

# 5️⃣ Summary
print("✅ Risk Classification Complete")
print(df["risk_level"].value_counts())

# 6️⃣ PCA Visualization
X = df.drop(columns=["decision_score", "anomaly_score", "risk_level"])

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)

# Map risk levels to numbers for coloring
risk_map = {"Low Risk": 0, "Medium Risk": 1, "High Risk": 2}
colors = df["risk_level"].map(risk_map)

plt.figure(figsize=(10, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.6)
plt.title("PCA Visualization with Risk Levels")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# 7️⃣ Save results
df.to_csv("transactions_with_risk_levels.csv", index=False)
print("✅ Saved: transactions_with_risk_levels.csv")

# 8️⃣ Save model + thresholds
model_package = {
    "model": iso,
    "low_threshold": low_threshold,
    "high_threshold": high_threshold
}

joblib.dump(model_package, "isolation_forest_with_risk.pkl")
print("✅ Saved: isolation_forest_with_risk.pkl")
import matplotlib.pyplot as plt

# Count risk levels
risk_counts = df["risk_level"].value_counts()

# Ensure correct order
order = ["Low Risk", "Medium Risk", "High Risk"]
risk_counts = risk_counts.reindex(order)

# Define professional colors
colors = ["green", "gold", "red"]

plt.figure(figsize=(7,5))
bars = plt.bar(risk_counts.index, risk_counts.values, color=colors)

plt.title("Distribution of Transaction Risk Levels")
plt.xlabel("Risk Level")
plt.ylabel("Number of Transactions")

# Add numbers above bars
for i in range(len(risk_counts)):
    plt.text(i, risk_counts.values[i] + 5, str(risk_counts.values[i]), ha='center')

plt.show()