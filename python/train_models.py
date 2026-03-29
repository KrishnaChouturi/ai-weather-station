import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.read_csv("weather_clean.csv")

df["time"] = pd.to_datetime(df["time"])
df["month"] = df["time"].dt.month
df["pressure_change"] = df["pressure"].diff().fillna(0)

features = ["temperature", "humidity", "pressure", "month", "pressure_change"]
X = df[features]
y = df["rain"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2%}")
print(f"F1 Score: {f1:.2f}")
print(classification_report(y_test, y_pred, target_names=["no rain", "rain"]))

# show which features the model found most important
importances = pd.Series(model.feature_importances_, index=features)
print("\nFeature importances:")
print(importances.sort_values(ascending=False))

with open("models/modelA_static.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/modelB_adaptive.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModels saved to models/")