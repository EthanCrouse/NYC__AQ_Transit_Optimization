import joblib
import pandas as pd

def grab_model(model_path="./air_quality_rf.joblib"):
    return joblib.load(model_path)

def grab_recent_measurements(month, year, lags=6, threshold=10):
    url = f"https://raw.githubusercontent.com/nychealth/nyccas-data/refs/heads/main/hist/csv/{year}/{month}.csv"
    df = pd.read_csv(url)
    # change time and site variables for analysis
    df['ObservationTimeUTC'] = pd.to_datetime(df['ObservationTimeUTC'], errors='coerce')

    # grab labels and add them with merge
    labels = pd.read_csv("data/AQ_2024/labels.csv")
    df = df.merge(labels, on="SiteID", how="left")
    df = df.dropna()

    # build lag features + target T/F
    def add_lags_and_target(g):
        for i in range(1, lags):
            g[f'lag_{i}'] = g['Value'].shift(i)
        g['target'] = (g['Value'] > threshold).astype('Int64')
        return g

    df = df.groupby('SiteName', group_keys=False).apply(add_lags_and_target)

    # drop incomplete rows
    lag_cols = [f'lag_{i}' for i in range(1, lags)]
    df = df.dropna(subset=lag_cols + ['target'])

    # stats across all lag columns
    df['lag_mean']   = df[lag_cols].mean(axis=1)
    df['lag_std']    = df[lag_cols].std(axis=1)
    df['lag_median'] = df[lag_cols].median(axis=1)

    # keep only the most recent reading per station
    df = df.sort_values("ObservationTimeUTC").groupby("SiteName", as_index=False).tail(1)
    return df


def predict_current(month, year, model=grab_model(), threshold=0.4):
    recent = grab_recent_measurements(month, year)
    lag_cols = [f"lag_{i}" for i in range(1, 6)] + ["lag_mean"] + ["lag_median"] + ["lag_std"]
    feature_cols = lag_cols + ["SiteName"]

    probs = model.predict_proba(recent[feature_cols])[:, 1]
    recent["BadAir"] = (probs >= threshold).astype(int)
    df = recent[["SiteName", "BadAir"]]
    return df


month=10
year=2024
df = predict_current(month, year)
print(df)


