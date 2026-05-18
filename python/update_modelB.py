import pandas as pd
import pickle
import glob
import numpy as np
import shutil
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

shutil.copy("models/modelB_adaptive.pkl", "models/modelB_adaptive_backup.pkl")
print("Model B backed up")

files = sorted(glob.glob("local_data/carmel_week*.csv"))
if not files:
    print("No local data found - nothing to update")
    exit()

hist = pd.read_csv("weather_clean.csv")
hist["time"] = pd.to_datetime(hist["time"])
hist = hist[hist["time"].dt.month.isin([3, 4, 5, 6, 7])]

for lag in [1, 2, 3]:
    hist[f"pressure_lag_{lag}"] = hist["pressure"].shift(lag)
    hist[f"hum_lag_{lag}"] = hist["humidity"].shift(lag)

hist["month"] = hist["time"].dt.month
hist["hour_val"] = hist["time"].dt.hour
hist["pressure_change"] = hist["pressure"].diff().fillna(0)
hist = hist.rename(columns={"time": "timestamp"})

session_starts = {
    "carmel_week1.csv": pd.Timestamp("2026-04-12 20:14"),
    "carmel_week2.csv": pd.Timestamp("2026-04-19 18:23"),
    "carmel_week3.csv": pd.Timestamp("2026-04-26 21:43"),
    "carmel_week4.csv": pd.Timestamp("2026-05-03 16:37"),
    "carmel_week5.csv": pd.Timestamp("2026-05-10 17:45")
}

def uptime_to_timestamp(uptime_str, session_start):
    try:
        parts = uptime_str.replace("h", "").replace("m", "").replace("s", "").split()
        total_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return session_start + pd.Timedelta(seconds=total_seconds)
    except:
        return pd.NaT

all_local_frames = []
for f in files:
    df = pd.read_csv(f)
    filename = os.path.basename(f)
    if filename in session_starts:
        start = session_starts[filename]
        df["timestamp"] = df["timestamp"].apply(lambda x: uptime_to_timestamp(x, start))
        df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce") - 4.0
        df["humidity_pct"] = pd.to_numeric(df["humidity_pct"], errors="coerce").clip(0, 100)
        df["rainfall_mm"] = df["rainfall_mm"].clip(upper=10.0) # Added hardware clip
        all_local_frames.append(df)

local = pd.concat(all_local_frames, ignore_index=True).dropna(subset=["timestamp"])
local["pressure_hpa"] = pd.to_numeric(local["pressure_hpa"], errors="coerce")
local = local.dropna(subset=["temperature_c", "humidity_pct", "pressure_hpa"])

local["hour"] = local["timestamp"].dt.floor("h")
local_hourly = local.groupby("hour").agg(
    temperature=("temperature_c", "mean"),
    humidity=("humidity_pct", "mean"),
    pressure=("pressure_hpa", "mean"),
    rain=("rainfall_mm", "sum")
).reset_index()

# Local Memory Features
for lag in [1, 2, 3]:
    local_hourly[f"pressure_lag_{lag}"] = local_hourly["pressure"].shift(lag)
    local_hourly[f"hum_lag_{lag}"] = local_hourly["humidity"].shift(lag)

local_hourly["rain"] = (local_hourly["rain"] >= 0.8382).astype(int)
local_hourly["month"] = local_hourly["hour"].dt.month
local_hourly["hour_val"] = local_hourly["hour"].dt.hour
local_hourly["pressure_change"] = local_hourly["pressure"].diff().fillna(0)
local_hourly = local_hourly.rename(columns={"hour": "timestamp"}).dropna()

# Updated feature list
features = ["temperature", "humidity", "pressure", "month", "pressure_change", "hour_val",
            "pressure_lag_1", "pressure_lag_2", "pressure_lag_3",
            "hum_lag_1", "hum_lag_2", "hum_lag_3"]

hist = hist.dropna()
hist_X = hist[features]
hist_y = hist["rain"]

local_X = local_hourly[features]
local_y = local_hourly["rain"]

LX_train, LX_test, Ly_train, Ly_test = train_test_split(local_X, local_y, test_size=0.2, random_state=42)

X_comb = pd.concat([hist_X, LX_train], ignore_index=True)
y_comb = pd.concat([hist_y, Ly_train], ignore_index=True)

weights = np.concatenate([np.ones(len(hist_X)), np.ones(len(LX_train)) * 4])

modelB = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42)
modelB.fit(X_comb, y_comb, sample_weight=weights)

y_pred = modelB.predict(LX_test)
print(f"Model B Updated with 4x weight")
print(f"Honest Accuracy (Unseen Local Data): {accuracy_score(Ly_test, y_pred):.2%}")
print(classification_report(Ly_test, y_pred))

with open("models/modelB_adaptive.pkl", "wb") as f:
    pickle.dump(modelB, f)