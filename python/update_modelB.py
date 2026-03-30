import pandas as pd
import pickle
import glob
import numpy as np
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# backup current Model B before overwriting
shutil.copy("models/modelB_adaptive.pkl", "models/modelB_adaptive_backup.pkl")
print("Model B backed up")

with open("models/modelB_adaptive.pkl", "rb") as f:
    modelB = pickle.load(f)

files = sorted(glob.glob("local_data/carmel_week*.csv"))
if not files:
    print("No local data found in local_data/ - nothing to update")
    exit()

print(f"Found {len(files)} week(s) of local data")

hist = pd.read_csv("weather_clean.csv")
hist["time"] = pd.to_datetime(hist["time"])
hist = hist[hist["time"].dt.month.isin([3, 4, 5, 6, 7])]
hist["month"] = hist["time"].dt.month
hist["pressure_change"] = hist["pressure"].diff().fillna(0)
hist = hist.rename(columns={"time": "timestamp"})

# IMPORTANT: every Sunday update that week's time to when plugged in the station
session_starts = {
    "carmel_week1.csv": pd.Timestamp("2026-04-05 09:00"),
    "carmel_week2.csv": pd.Timestamp("2026-04-12 09:00"),
    "carmel_week3.csv": pd.Timestamp("2026-04-19 09:00"),
    "carmel_week4.csv": pd.Timestamp("2026-04-26 09:00"),
    "carmel_week5.csv": pd.Timestamp("2026-05-03 09:00"),
    "carmel_week6.csv": pd.Timestamp("2026-05-10 09:00"),
    "carmel_week7.csv": pd.Timestamp("2026-05-17 09:00"),
    "carmel_week8.csv": pd.Timestamp("2026-05-24 09:00"),
    "carmel_week9.csv": pd.Timestamp("2026-05-31 09:00"),
    "carmel_week10.csv": pd.Timestamp("2026-06-07 09:00"),
    "carmel_week11.csv": pd.Timestamp("2026-06-14 09:00"),
    "carmel_week12.csv": pd.Timestamp("2026-06-21 09:00"),
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
    filename = f.split("/")[-1].split("\\")[-1]
    if filename not in session_starts:
        print(f"Warning: no session start time for {filename}, skipping")
        continue
    session_start = session_starts[filename]
    df["timestamp"] = df["timestamp"].apply(lambda x: uptime_to_timestamp(x, session_start))
    all_local_frames.append(df)

local = pd.concat(all_local_frames, ignore_index=True)
local = local.dropna(subset=["timestamp"])

local["temperature_c"] = pd.to_numeric(local["temperature_c"], errors="coerce")
local["humidity_pct"] = pd.to_numeric(local["humidity_pct"], errors="coerce")
local["pressure_hpa"] = pd.to_numeric(local["pressure_hpa"], errors="coerce")
local = local.dropna(subset=["temperature_c", "humidity_pct", "pressure_hpa"])

local["hour"] = local["timestamp"].dt.floor("h")
local_hourly = local.groupby("hour").agg(
    temperature=("temperature_c", "mean"),
    humidity=("humidity_pct", "mean"),
    pressure=("pressure_hpa", "mean"),
    rain=("rainfall_mm", "sum")
).reset_index()

local_hourly["rain"] = (local_hourly["rain"] > 0).astype(int)
local_hourly["month"] = local_hourly["hour"].dt.month
local_hourly["pressure_change"] = local_hourly["pressure"].diff().fillna(0)
local_hourly = local_hourly.rename(columns={"hour": "timestamp"})

features = ["temperature", "humidity", "pressure", "month", "pressure_change"]

hist_X = hist[features].dropna()
hist_y = hist.loc[hist_X.index, "rain"]

local_X = local_hourly[features].dropna()
local_y = local_hourly.loc[local_X.index, "rain"]

X_combined = pd.concat([hist_X, local_X], ignore_index=True)
y_combined = pd.concat([hist_y, local_y], ignore_index=True)

# local data weighted 5x
hist_weights = np.ones(len(hist_X))
local_weights = np.ones(len(local_X)) * 5
weights_combined = np.concatenate([hist_weights, local_weights])

# retrain Model B
modelB = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)
modelB.fit(X_combined, y_combined, sample_weight=weights_combined)

y_local_pred = modelB.predict(local_X)
local_accuracy = accuracy_score(local_y, y_local_pred)

# save updated Model B
with open("models/modelB_adaptive.pkl", "wb") as f:
    pickle.dump(modelB, f)

print(f"Model B updated with {len(local_X)} local hours + {len(hist_X)} historical hours")
print(f"Local data weighted 5x")
print(f"Model B accuracy on local data: {local_accuracy:.2%}")
