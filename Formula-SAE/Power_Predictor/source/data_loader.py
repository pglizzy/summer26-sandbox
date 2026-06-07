import pandas as pd
from .paths import find_data_file


def load_engine_specs() -> pd.DataFrame:
    path = find_data_file("COMPARISONS_RotatingAssy.csv")
    return pd.read_csv(path)


def load_dyno_curves() -> pd.DataFrame:
    path = find_data_file("COMPARISONS_dynocurves.csv")
    return pd.read_csv(path)


def load_fuels() -> pd.DataFrame:
    path = find_data_file("FUELS.csv")
    return pd.read_csv(path)


def load_transmissions() -> pd.DataFrame:
    path = find_data_file("COMPARISONS_Reductions.csv")
    return pd.read_csv(path)