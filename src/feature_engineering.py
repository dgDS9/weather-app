import numpy as np
import pandas as pd

from config import HORIZON_HOURS, TARGET


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df["dt_iso"].dt.hour
    df["day_of_year"] = df["dt_iso"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET] = df["temp"].shift(-HORIZON_HOURS)
    return df


def add_temperature_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for lag in [1, 3, 6, 12, 24]:
        df[f"temp_lag_{lag}"] = df["temp"].shift(lag)
    return df


def add_extended_weather_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the same meteorological lag candidates used in the notebook.
    The final model intentionally selects only a subset of them.
    """
    df = df.copy()

    weather_columns = [
        "humidity",
        "pressure",
        "wind_speed",
        "wind_deg",
        "clouds_all",
    ]

    for column in weather_columns:
        if column not in df.columns:
            continue
        for lag in [1, 6, 24]:
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)

    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Complete feature engineering used by training/evaluation."""
    df = add_time_features(df)
    df = add_target(df)
    df = add_temperature_lags(df)
    df = add_extended_weather_lags(df)
    return df


if __name__ == "__main__":
    from data_preparation import load_and_prepare_data
    from config import EXTENDED_FEATURES, TARGET

    data = build_feature_table(load_and_prepare_data())
    print(data[EXTENDED_FEATURES + [TARGET, "dt_iso"]].dropna().head())
