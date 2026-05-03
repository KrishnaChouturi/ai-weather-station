import pandas as pd
import pickle
import os

with open("models/modelA_static.pkl", "rb") as f:
    modelA = pickle.load(f)

with open("models/modelB_adaptive.pkl", "rb") as f:
    modelB = pickle.load(f)

# Change filename here each week
df = pd.read_csv("local_data/carmel_week3.csv")
session_start = pd.Timestamp("2026-04-26 21:43")

def uptime_to_timestamp(uptime_str):
    parts = uptime_str.replace("h", "").replace("m", "").replace("s", "").split()
    total_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return session_start + pd.Timedelta(seconds=total_seconds)

df["timestamp"] = df["timestamp"].apply(uptime_to_timestamp)
df["temperature_c"] = df["temperature_c"] - 4.0
df["humidity_pct"] = df["humidity_pct"].clip(0, 100)
df["hour"] = df["timestamp"].dt.floor("h")

hourly = df.groupby("hour").agg(
    temperature=("temperature_c", "mean"),
    humidity=("humidity_pct", "mean"),
    pressure=("pressure_hpa", "mean"),
    actual_rain=("rainfall_mm", "sum")
).reset_index()

hourly["actual_rain"] = (hourly["actual_rain"] >= 0.8382).astype(int)
hourly["month"] = hourly["hour"].dt.month
hourly["hour_val"] = hourly["hour"].dt.hour
hourly["pressure_change"] = hourly["pressure"].diff().fillna(0)
hourly = hourly.dropna()

features = ["temperature", "humidity", "pressure", "month", "pressure_change", "hour_val"]
hourly["prediction_A"] = modelA.predict(hourly[features])
hourly["prediction_B"] = modelB.predict(hourly[features])

output = hourly[["hour", "actual_rain", "prediction_A", "prediction_B"]]
output = output.rename(columns={"hour": "timestamp"})

log_path = "predictions_log.csv"
if os.path.exists(log_path):
    output.to_csv(log_path, mode="a", header=False, index=False)
else:
    output.to_csv(log_path, index=False)

print(f"Predictions saved - {len(output)} hours processed")