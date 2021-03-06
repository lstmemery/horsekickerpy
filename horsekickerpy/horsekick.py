import warnings

import pandas as pd


class MissingColumnException(Exception):
    pass


class YearOutBoundsWarning(Warning):
    pass


class UnknownCorpsWarning(Warning):
    pass


def clean_horse_kicks(horse_kick_df: pd.DataFrame) -> pd.DataFrame:
    prussian_corps = {"G", "I", "II", "III", "IV", "V", "VI",
                         "VII", "VIII", "IX", "X", "XI", "XIV", "XV"}

    if not {"year", "corps"}.issubset(horse_kick_df.columns):
        missing_columns = {"year", "corps"} - set(horse_kick_df.columns)
        raise MissingColumnException(f"{missing_columns} not in Data Frame.")

    years_in_bounds = horse_kick_df["year"].between(1701, 1919, inclusive=True)
    if not all(years_in_bounds):
        out_of_bounds_years = horse_kick_df.loc[years_in_bounds, "year"]
        warnings.warn(
            f"Years out of bounds: {out_of_bounds_years}",
            YearOutBoundsWarning,
        )

    corps_known = horse_kick_df["corps"].isin(prussian_corps)
    if not all(horse_kick_df["corps"].isin(prussian_corps)):
        unknown_corps = prussian_corps - set(horse_kick_df["corps"])
        warnings.warn(
            f"Unknown Corps: {unknown_corps}",
            UnknownCorpsWarning
        )

    return horse_kick_df.loc[
        corps_known & years_in_bounds,
        ["year", "corps"]
    ]
