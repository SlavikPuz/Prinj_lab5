import pandas as pd


def survival_percent(df: pd.DataFrame) -> float:
    n = len(df)
    if n == 0:
        return 0.0
    return float(df["Survived"].sum()) / n * 100.0


def young_old_survival_by_class(
    df: pd.DataFrame, pclass: int
) -> tuple[float, float]:
    df_class = df[df["Pclass"] == pclass]
    young = df_class[df_class["Age"] < 30]
    old = df_class[df_class["Age"] > 60]
    return survival_percent(young), survival_percent(old)
