from typing import NamedTuple, Callable, TypeVar, Union
import dill
import numpy as np
import pandas as pd
from scipy.stats import linregress

# TODO: update mypy to a version with https://github.com/python/mypy/pull/14041
_Self = TypeVar("_Self")
_NumericType = TypeVar(
    "_NumericType", bound=Union[int, float, complex, str, bytes, np.generic]
)
_VppStdModel = Callable[[_NumericType], _NumericType]


def _get_player_averages(performances: pd.DataFrame) -> pd.DataFrame:
    """Get career averages for all players,
    limiting to only players with reasonable sample sizes
    becuase I'm using this for fitting the game | career model"""
    by_player = (
        performances.groupby("player_id")
        .agg({"value": "sum", "n_possessions": "sum", "game_id": "count"})
        .reset_index()
        .rename(
            columns={
                "value": "total_value",
                "n_possessions": "total_possessions",
                "game_id": "n_games",
            }
        )
    )

    # Reasonable # of games
    by_player = by_player[by_player.n_games > 15]
    # Reasonable number of possessions
    return by_player[by_player.total_possessions > 100]


def add_player_aggregates(
    performances: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add columns to the df representing
    their career averages from all other games"""
    by_player = _get_player_averages(performances)
    return (
        performances.merge(by_player, on="player_id")
        .assign(
            value_without_this_game=lambda _: _.total_value - _["value"],
            possessions_without_this_game=lambda _: _.total_possessions
            - _.n_possessions,
        )
        .assign(
            other_game_vpp=lambda _: _.value_without_this_game
            / _.possessions_without_this_game,
            vpp=lambda _: _["value"] / _.n_possessions,
        )
        .assign(vpp_diff=lambda _: _.vpp - _.other_game_vpp)
    ), by_player


def fit_std_by_sample_size(
    performances: pd.DataFrame,
) -> tuple[pd.DataFrame, _VppStdModel]:
    """Make a model that predicts sd of VPP given sample size"""
    std_by_sample_size = (
        performances.assign(
            percentile=lambda _: (
                np.argsort(_.n_possessions).argsort() / len(_) * 100
            ).astype(int)
        )
        .groupby("percentile")
        .agg({"vpp": "std", "n_possessions": "median"})
        .reset_index()
        .rename(columns={"vpp": "value_std", "n_possessions": "median_possessions"})
        .assign(inv_value_std=lambda _: 1 / (_.value_std**3))
    )
    regression = linregress(
        std_by_sample_size.median_possessions, std_by_sample_size.inv_value_std
    )

    def get_vpp_sd(n_poss: _NumericType) -> _NumericType:
        n_poss = np.maximum(n_poss, std_by_sample_size.median_possessions.min())
        n_poss = np.minimum(n_poss, std_by_sample_size.median_possessions.max())
        predicted_inverse = regression.intercept + regression.slope * n_poss
        return 1 / np.power(predicted_inverse, 1 / 3)

    return std_by_sample_size, get_vpp_sd


class LeagueModel(NamedTuple):
    """Everything we need to store for a league's update parameters"""

    possessions_to_vpp_std: _VppStdModel
    vpp_mean: float
    vpp_variance: float

    def save(self, filename: str):
        """Store the league params as a file"""
        with open(filename, "wb") as file:
            dill.dump(self, file)

    @classmethod
    def load(cls: type[_Self], filename: str) -> _Self:
        """Read league params from a pickle file"""
        with open(filename, "rb") as file:
            return dill.load(file)
