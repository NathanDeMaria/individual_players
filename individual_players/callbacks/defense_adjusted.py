from collections import defaultdict
import pandas as pd

from individual_players.priors.types import Player

from .types import TeamCallback, PlayerCallback
from ..types import RatingsLookup
from ..ratings import PlayerRatings
from ..league_model import LeagueModel


class DefenseAdjustedCallback:
    def __init__(
        self,
        defense_model: LeagueModel,
        offense_adjusted_model: LeagueModel,
    ):
        self._defense_model = defense_model

        self._defensive_performances: dict[str, float] = {}

        self._defense_ratings: RatingsLookup = defaultdict(
            lambda: (defense_model.vpp_mean, defense_model.vpp_variance)
        )
        self._defense_adjustment: dict[str, float] = {}

        self._adjusted_offense_rating: RatingsLookup = defaultdict(
            lambda: (
                offense_adjusted_model.vpp_mean,
                offense_adjusted_model.vpp_variance,
            )
        )

    @property
    def adjusted_offense_rating(self) -> RatingsLookup:
        return self._adjusted_offense_rating

    @property
    def defense_ratings(self) -> RatingsLookup:
        return self._defense_ratings

    @property
    def team_callback(self) -> TeamCallback:
        def store_defense_adjustment(
            _: str, player_ratings: RatingsLookup, team: pd.DataFrame
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
            self._update_adjusted_offensive_rating(player_performance)

        return store_result

    @property
    def defensive_performances(self) -> pd.DataFrame:
        return pd.DataFrame(self._defensive_performances).rename(
            columns={"opponent_vs_expectation": "value"}
        )

    def _update_defense_rating(self, player_performance):
        opp_performance = self._defensive_performances[player_performance.opponent_id]
        defense_mu, defense_var = self._defense_ratings[player_performance.player_id]
        self._defense_ratings[player_performance.player_id] = _update_rating(
            defense_mu, defense_var, opp_performance, player_performance.defense_sd
        )

    def _get_defensive_adjustment(self, opponent_id: str, team) -> float:
        total_possessions = team.n_possessions.sum()
        weighted_adjustment = (
            sum(
                self._defense_ratings[p.player_id][0] * p.n_possessions
                for p in team.itertuples()
            )
            / total_possessions
        )
        self._defense_adjustment[opponent_id] = weighted_adjustment

    def _update_adjusted_offensive_rating(self, player_performance):
        rating = self._adjusted_offense_rating[player_performance.player_id]
        defense_adjustment = self._defense_adjustment[player_performance.opponent_id]

        # TODO: double check the sign here
        adjusted_performance = (
            player_performance.value / player_performance.n_possessions
            - defense_adjustment
        )
        self._adjusted_offense_rating[player_performance.player_id] = _update_rating(
            *rating,
            adjusted_performance,
            player_performance.adjusted_vpp_sd,
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
        sum(player_ratings.get_rating(Player(p.player_id, p.team_id))[0] * p.n_possessions for p in team.itertuples())
        / team.n_possessions.sum()
    )
    return vpp - expected_vpp
