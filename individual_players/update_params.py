from dataclasses import dataclass
from typing import List, Tuple, NamedTuple

import pandas as pd
import numpy as np
from dataclasses_json import DataClassJsonMixin
from sklearn import linear_model


_VALUE_COLUMN = "value"
_POSSESSIONS_COLUMN = "possessions"
_VPP_COLUMN = "vpp"
_CAREER_SUFFIX = "_career"
_VPP_CAREER_COLUMN = f"{_VPP_COLUMN}{_CAREER_SUFFIX}"


@dataclass
class UpdateParams(DataClassJsonMixin):
    """
    Get the update parameters relevant for a bayesian update
    of a player's career VPP given one game's results

    possession_std_coefficients
        The coefficients in a LM that gets the STD
        of an individual game's VPP for a player
        given the # of possessions.
    """

    career_mean: float
    career_std: float
    possession_std_coefficients: List[float]


class _FitResult(NamedTuple):
    coefficients: List[float]
    data: np.ndarray
    x: List[float]
    y: List[float]

    @property
    def y_hat(self) -> np.ndarray:
        return np.matmul(self.data, np.array(self.coefficients))


def _fit_vpp_std(values_df: pd.DataFrame) -> _FitResult:
    """
    Fit the STD of VPP for a single game as a polynomial
    """
    # Why these breaks?
    # Because I want to ignore games with fewer than 10 possessions
    # and 110 is a reasonable max
    breaks = np.arange(10, 105, 5)
    min_possessions = []
    vpp_std = []
    for i in range(len(breaks) - 1):
        start, end = breaks[i : i + 2]
        game_vs_career = values_df[_VPP_COLUMN] - values_df[_VPP_CAREER_COLUMN]
        game_vs_career = game_vs_career[
            np.logical_and(
                values_df[_POSSESSIONS_COLUMN] >= start,
                values_df[_POSSESSIONS_COLUMN] < end,
            )
        ]
        game_vs_career = game_vs_career[np.isfinite(game_vs_career)]
        min_possessions.append(start)
        vpp_std.append(game_vs_career.std())

    regression = linear_model.LinearRegression()
    data = np.stack([np.power(min_possessions, i) for i in range(4)], axis=1)
    regression.fit(data[:, 1:], vpp_std)
    return _FitResult(
        coefficients=[regression.intercept_, *regression.coef_],
        data=data,
        y=vpp_std,
        x=min_possessions,
    )


def fit_params(
    values_csv_path: str,
) -> Tuple[UpdateParams, pd.DataFrame, pd.DataFrame, _FitResult]:
    """
    Build parameters we'll use in "update" scripts
    """
    values_df = pd.read_csv(values_csv_path)
    values_df["vpp"] = values_df[_VALUE_COLUMN] / values_df[_POSSESSIONS_COLUMN]

    # Add in the career average VPP to each row
    career_averages = (
        values_df.groupby("player")
        .agg("sum")[[_VALUE_COLUMN, _POSSESSIONS_COLUMN]]
        .reset_index()
        .reset_index()
    )
    career_averages["vpp"] = (
        career_averages[_VALUE_COLUMN] / career_averages[_POSSESSIONS_COLUMN]
    )
    career_averages = career_averages[career_averages[_POSSESSIONS_COLUMN] > 100]
    values_df = values_df.merge(
        career_averages, on="player", suffixes=("", _CAREER_SUFFIX)
    )

    fit_result = _fit_vpp_std(values_df)

    params = UpdateParams(
        career_mean=career_averages[_VPP_COLUMN].mean(),
        career_std=career_averages[_VPP_COLUMN].std(),
        possession_std_coefficients=fit_result.coefficients,
    )
    return params, career_averages, values_df, fit_result
