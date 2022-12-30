from collections import defaultdict
from typing import NamedTuple, Callable, TypeVar, Union
import dill
import numpy as np
import pandas as pd
import statsmodels.api as sm

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
        .agg(
            {
                "value": [("total_value", "sum")],
                "n_possessions": [("n_games", "count"), ("total_possessions", "sum")],
            }
        )
        .reset_index()
    )
    other_columns = by_player.columns.get_level_values(1).tolist()[1:]
    by_player.columns = ["player_id"] + other_columns

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


def add_opponent(performances: pd.DataFrame) -> pd.DataFrame:
    """Add a column for opp_team_id"""
    unique_games = performances[["game_id", "team_id"]].drop_duplicates()
    game_teams = defaultdict(list)
    for row in unique_games.itertuples():
        game_teams[row.game_id].append(row.team_id)
    assert all(len(teams) == 2 for teams in game_teams.values())

    game_team_df = (
        pd.DataFrame(game_teams)
        .T.reset_index()
        .rename(columns={0: "team_id", 1: "opponent_id", "index": "game_id"})
    )
    other_half = game_team_df.rename(
        columns={"team_id": "opponent_id", "opponent_id": "team_id"}
    )
    game_team_df = pd.concat([game_team_df, other_half], axis=0)

    return performances.merge(game_team_df, on=["game_id", "team_id"])


def fit_std_by_sample_size(
    performances: pd.DataFrame,
    inv_power: int = 3,
    polynomial_power: int = 1,
) -> tuple[pd.DataFrame, _VppStdModel]:
    """Make a model that predicts sd of VPP given sample size

    Parameters
    ----------
    performances
        Data frame of game performances
    inv_power, optional
        power that makes the std linear (when flipped negative)
        Try out a few values and plot it
    polynomial_power, optional
        power of the linear model (ex: 2 makes it fit y = a + b * x + c * x2)
    """

    def _build_features(x):
        return np.column_stack([np.power(x, i) for i in range(polynomial_power + 1)])

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
        .assign(inv_value_std=lambda _: np.power(_.value_std, -inv_power))
    )

    features = _build_features(std_by_sample_size.median_possessions)
    regression = sm.OLS(std_by_sample_size.inv_value_std, features).fit()

    def get_vpp_sd(n_poss: _NumericType) -> _NumericType:
        n_poss = np.clip(
            n_poss,
            a_min=std_by_sample_size.median_possessions.min(),
            a_max=std_by_sample_size.median_possessions.max(),
        )
        features = _build_features(n_poss)
        predicted_inverse = np.matmul(features, regression.params)
        return 1 / np.power(predicted_inverse, 1 / inv_power)

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
