from collections import defaultdict
from typing import NamedTuple
import pandas as pd

from .types import TeamCallback, PlayerCallback
from ..types import RatingsLookup
from ..ratings import PlayerRatings, Player
from ..league_model import LeagueModel
from ..priors import get_simple_prior, PriorGetter


class _DefensiveAdjustedPerformance(NamedTuple):
    player_id: str
    n_possessions: float
    adjusted_performance: float


class DefenseAdjustingCallback:
    def __init__(
        self,
        defense_model: LeagueModel,
        prior_getter: PriorGetter | None = None,
    ):
        self._defense_model = defense_model

        self._defensive_performances: dict[str, float] = {}

        if prior_getter is None:
            prior_getter = get_simple_prior(defense_model)

        self._defense_ratings = PlayerRatings(prior_getter)
        self._defense_adjustment: dict[str, float] = {}

        self._adjusted_offensive_performances: list[_DefensiveAdjustedPerformance] = []

    @property
    def defense_ratings(self) -> PlayerRatings:
        return self._defense_ratings

    @property
    def team_callback(self) -> TeamCallback:
        def store_defense_adjustment(
            _: str, player_ratings: PlayerRatings, team: pd.DataFrame
        ):
            opponent_id = _get_opponent_id(team)
            self._defensive_performances[opponent_id] = _get_defensive_difference(
                team, player_ratings
            )
            self._get_defensive_adjustment(opponent_id, team)

        return store_defense_adjustment

    @property
    def player_callback(self) -> PlayerCallback:
        def store_result(_: str, player_performance):
            self._update_defense_rating(player_performance)
            self._store_adjusted_offensive_performance(player_performance)

        return store_result

    @property
    def defensive_performances(self) -> pd.DataFrame:
        # TODO: can I stop exposing some of these?
        # Or maybe inherit as I build "layers?"
        return pd.DataFrame(self._defensive_performances).rename(
            columns={"opponent_vs_expectation": "value"}
        )

    def _update_defense_rating(self, player_performance):
        opp_performance = self._defensive_performances[player_performance.opponent_id]
        defense_mu, defense_var = self._defense_ratings.get_rating(Player(player_performance.player_id, player_performance.team_id))
        new_rating = _update_rating(
            defense_mu, defense_var, opp_performance, player_performance.defense_sd
        )
        self._defense_ratings.update_rating(player_performance.player_id, new_rating)

    def _get_defensive_adjustment(self, opponent_id: str, team):
        total_possessions = team.n_possessions.sum()
        weighted_adjustment = (
            sum(
                self._defense_ratings.get_rating(Player(p.player_id, p.team_id))[0] * p.n_possessions
                for p in team.itertuples()
            )
            / total_possessions
        )
        self._defense_adjustment[opponent_id] = weighted_adjustment

    def _store_adjusted_offensive_performance(self, player_performance):
        defense_adjustment = self._defense_adjustment[player_performance.opponent_id]
        adjusted_performance = (
            player_performance.value / player_performance.n_possessions
            - defense_adjustment
        )
        self._adjusted_offensive_performances.append(
            _DefensiveAdjustedPerformance(
                n_possessions=player_performance.n_possessions,
                player_id=player_performance.player_id,
                adjusted_performance=adjusted_performance,
            )
        )

    @property
    def adjusted_performances(self) -> pd.DataFrame:
        return pd.DataFrame(self._adjusted_offensive_performances).rename(
            columns={"adjusted_performance": "value"}
        )


# TODO: DRY with the loop?3
def _update_rating(
    player_mu: float, player_var: float, game_value: float, value_std: float
) -> tuple[float, float]:
    # Rating Update
    new_mu = (player_mu * (value_std**2) + game_value * player_var) / (
        (value_std**2) + player_var
    )
    new_var = 1 / ((1 / player_var + 1 / (value_std**2)))
    return new_mu, new_var


def _get_opponent_id(team) -> str:
    opponent_ids = team.opponent_id.unique()
    assert len(opponent_ids) == 1
    return opponent_ids.item()


def _get_defensive_difference(team, player_ratings: PlayerRatings) -> float:
    vpp = team.value.sum() / team.n_possessions.sum()
    expected_vpp = (
        sum(
            player_ratings.get_rating(Player(p.player_id, p.team_id))[0]
            * p.n_possessions
            for p in team.itertuples()
        )
        / team.n_possessions.sum()
    )
    return vpp - expected_vpp
