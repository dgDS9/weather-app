import pandas as pd

from config import DATA_PATH

NUMERIC_COLUMNS = [
    "temp",
    "dew_point",
    "pressure",
    "humidity",
    "wind_speed",
    "wind_deg",
    "clouds_all",
]


def load_and_prepare_data(csv_path=DATA_PATH) -> pd.DataFrame:
    """Load raw OpenWeather CSV, normalize dtypes and sort chronologically."""
    df = pd.read_csv(csv_path, sep=",", dtype=str)

    df["dt"] = pd.to_numeric(df["dt"], errors="coerce")
    df["dt_iso"] = pd.to_datetime(
        df["dt_iso"],
        format="%Y-%m-%d %H:%M:%S %z UTC",
        errors="coerce",
    )

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=["dt_iso", "temp"])
        .sort_values("dt_iso")
        .reset_index(drop=True)
    )

    return df


if __name__ == "__main__":
    data = load_and_prepare_data()
    print(data.head())
    print(f"\nRows: {len(data):,}")
    print(f"From: {data['dt_iso'].min()}")
    print(f"To:   {data['dt_iso'].max()}")
