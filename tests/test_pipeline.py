import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------
# src-Ordner für Tests verfügbar machen
# ------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))


from data_preparation import load_and_prepare_data
from feature_engineering import (
    add_target,
    add_temperature_lags,
    add_time_features,
    build_feature_table,
)
from evaluation import calculate_metrics
from training import temporal_split
from config import (
    EXTENDED_FEATURES,
    TARGET,
)


# ======================================================
# DATA PREPARATION
# ======================================================

def test_load_and_prepare_data(tmp_path):
    """
    Prüft:
    - CSV kann geladen werden
    - Temperatur wird numerisch
    - Daten werden chronologisch sortiert
    """

    csv_file = tmp_path / "weather.csv"

    csv_file.write_text(
        (
            "dt,dt_iso,temp,dew_point,pressure,humidity,"
            "wind_speed,wind_deg,clouds_all\n"
            "2,2020-01-01 02:00:00 +0000 UTC,12,5,1010,70,3,180,40\n"
            "1,2020-01-01 01:00:00 +0000 UTC,10,4,1012,75,2,170,30\n"
        ),
        encoding="utf-8",
    )

    df = load_and_prepare_data(csv_file)

    assert len(df) == 2

    # chronologische Sortierung
    assert df.iloc[0]["temp"] == 10
    assert df.iloc[1]["temp"] == 12

    # temp sollte numerisch sein
    assert pd.api.types.is_numeric_dtype(df["temp"])


# ======================================================
# TIME FEATURES
# ======================================================

def test_add_time_features():
    df = pd.DataFrame(
        {
            "dt_iso": pd.to_datetime(
                [
                    "2020-01-01 00:00:00+00:00",
                    "2020-01-01 06:00:00+00:00",
                ]
            )
        }
    )

    result = add_time_features(df)

    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns
    assert "doy_sin" in result.columns
    assert "doy_cos" in result.columns

    # Mitternacht:
    # sin(0) = 0
    assert np.isclose(
        result.iloc[0]["hour_sin"],
        0.0,
    )

    # Mitternacht:
    # cos(0) = 1
    assert np.isclose(
        result.iloc[0]["hour_cos"],
        1.0,
    )


# ======================================================
# TARGET t+24
# ======================================================

def test_add_target():
    df = pd.DataFrame(
        {
            "temp": np.arange(30, dtype=float)
        }
    )

    result = add_target(df)

    # Temperatur 24 Stunden später
    assert result.loc[0, TARGET] == 24.0
    assert result.loc[1, TARGET] == 25.0

    # Letzte 24 Zeilen können kein t+24 Target haben
    assert result[TARGET].isna().sum() == 24


# ======================================================
# LAG FEATURES
# ======================================================

def test_temperature_lags():
    df = pd.DataFrame(
        {
            "temp": np.arange(30, dtype=float)
        }
    )

    result = add_temperature_lags(df)

    assert result.loc[1, "temp_lag_1"] == 0
    assert result.loc[3, "temp_lag_3"] == 0
    assert result.loc[6, "temp_lag_6"] == 0
    assert result.loc[12, "temp_lag_12"] == 0
    assert result.loc[24, "temp_lag_24"] == 0


# ======================================================
# COMPLETE FEATURE ENGINEERING
# ======================================================

def test_build_feature_table():
    rows = 50

    df = pd.DataFrame(
        {
            "dt_iso": pd.date_range(
                "2020-01-01",
                periods=rows,
                freq="h",
                tz="UTC",
            ),
            "temp": np.arange(rows, dtype=float),
            "humidity": np.full(rows, 70.0),
            "pressure": np.full(rows, 1013.0),
            "wind_speed": np.full(rows, 3.0),
            "wind_deg": np.full(rows, 180.0),
            "clouds_all": np.full(rows, 50.0),
        }
    )

    result = build_feature_table(df)

    for feature in EXTENDED_FEATURES:
        assert feature in result.columns

    assert TARGET in result.columns


# ======================================================
# TEMPORAL SPLIT
# ======================================================

def test_temporal_split():
    df = pd.DataFrame(
        {
            "dt_iso": pd.to_datetime(
                [
                    "2017-01-01",
                    "2018-06-01",
                    "2020-06-01",
                ],
                utc=True,
            ),
            "temp": [10, 12, 14],
        }
    )

    train, validation, test = temporal_split(df)

    assert len(train) == 1
    assert len(validation) == 1
    assert len(test) == 1

    assert train.iloc[0]["temp"] == 10
    assert validation.iloc[0]["temp"] == 12
    assert test.iloc[0]["temp"] == 14


# ======================================================
# EVALUATION METRICS
# ======================================================

def test_calculate_metrics():
    y_true = np.array([10.0, 12.0])
    y_pred = np.array([11.0, 10.0])

    mae, rmse = calculate_metrics(
        y_true,
        y_pred,
    )

    # Fehler:
    # |10 - 11| = 1
    # |12 - 10| = 2
    #
    # MAE = (1 + 2) / 2 = 1.5

    assert np.isclose(mae, 1.5)

    # RMSE = sqrt((1² + 2²) / 2)
    expected_rmse = np.sqrt(2.5)

    assert np.isclose(
        rmse,
        expected_rmse,
    )