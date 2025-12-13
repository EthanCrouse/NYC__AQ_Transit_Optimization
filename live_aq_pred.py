import joblib
import pandas as pd
import urllib.error
import time
import os

def grab_model(model_path = os.path.join(os.path.dirname(__file__), "model.joblib")):
    return joblib.load(model_path)

def read_csv_with_retry(url, retries=3, delay=2):
    """
    Try to read a CSV from a URL, retrying on URLError/connection reset.
    """
    for attempt in range(1, retries + 1):
        try:
            return pd.read_csv(url)
        except urllib.error.URLError as e:
            print(f"[Attempt {attempt}/{retries}] Failed to read {url}: {e}")
            if attempt == retries:
                # re-raise on last attempt
                raise
            time.sleep(delay * attempt)

def grab_recent_measurements(month, year, lags=6, threshold=10, for_training=False):
    """
    If for_training=True:
        - returns many rows with a 'target' column that says whether the
          *next* reading is over the threshold.
    If for_training=False:
        - returns a single most-recent row per SiteName with lag features,
          suitable for predicting the *next* reading.
    """
    url = f"https://raw.githubusercontent.com/nychealth/nyccas-data/refs/heads/main/hist/csv/{year}/{month}.csv"
    
    df = read_csv_with_retry(url)
    df['ObservationTimeUTC'] = pd.to_datetime(df['ObservationTimeUTC'], errors='coerce')

    # grab labels and add them with merge
    labels = pd.read_csv("data/AQ_2024/labels.csv")
    df = df.merge(labels, on="SiteID", how="left")
    df = df.dropna(subset=["SiteName", "ObservationTimeUTC", "Value"])

    # build lag features + (optionally) "next-reading" target
    def add_lags_and_target(g):
        # sort by time within each site just in case
        g = g.sort_values("ObservationTimeUTC")

        for i in range(1, lags):
            g[f'lag_{i}'] = g['Value'].shift(i)

        if for_training:
            # next reading
            g['future_value'] = g['Value'].shift(-1)
            g['target'] = (g['future_value'] > threshold).astype('Int64')

        return g

    df = df.groupby('SiteName', group_keys=False).apply(add_lags_and_target)

    lag_cols = [f'lag_{i}' for i in range(1, lags)]

    if for_training:
        # we need full lags AND a known future (target)
        df = df.dropna(subset=lag_cols + ['target'])

        # stats across all lag columns
        df['lag_mean']   = df[lag_cols].mean(axis=1)
        df['lag_std']    = df[lag_cols].std(axis=1)
        df['lag_median'] = df[lag_cols].median(axis=1)

        return df
    else:
        # inference: we don't have future target, only use rows with full lags
        df = df.dropna(subset=lag_cols)

        # stats across all lag columns
        df['lag_mean']   = df[lag_cols].mean(axis=1)
        df['lag_std']    = df[lag_cols].std(axis=1)
        df['lag_median'] = df[lag_cols].median(axis=1)

        # keep only the most recent reading per station (to predict the NEXT one)
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

if __name__ == "__main__":
    month=3
    year=2025
    df = predict_current(month, year)
    print(df)


