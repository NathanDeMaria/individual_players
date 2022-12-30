from typing import NamedTuple
import pandas as pd

from .types import TeamCallback, PlayerCallback
from ..types import PlayerRatings


class _DefensivePerformance(NamedTuple):
    player_id: str
    n_possessions: float
    opponent_vs_expectation: float


class DefenseCallback:
    def __init__(self):
        # team -> VPP in most recent game
        # We're just banking on the callbacks being called back to back
        self._performances: dict[str, float] = {}
        self._defensive_performances: list[_DefensivePerformance] = []

    @property
    def team_callback(self) -> TeamCallback:
        def store_game_performance(
            _: str, player_ratings: PlayerRatings, team: pd.DataFrame
        ):
            opponent_ids = team.opponent_id.unique()
            assert len(opponent_ids) == 1
            opponent_id = opponent_ids.item()
            vpp = team.value.sum() / team.n_possessions.sum()
            expected_vpp = (
                sum(
                    player_ratings[p.player_id][0] * p.n_possessions
                    for p in team.itertuples()
                )
                / team.n_possessions.sum()
            )
            self._performances[opponent_id] = vpp - expected_vpp

        return store_game_performance

    @property
    def player_callback(self) -> PlayerCallback:
        def store_result(_: str, player_performance):
            opp_performance = self._performances[player_performance.opponent_id]
            self._defensive_performances.append(
                _DefensivePerformance(
                    player_id=player_performance.player_id,
                    n_possessions=player_performance.n_possessions,
                    opponent_vs_expectation=opp_performance,
                )
            )

        return store_result

    @property
    def defensive_performances(self) -> pd.DataFrame:
        return pd.DataFrame(self._defensive_performances).rename(
            columns={"opponent_vs_expectation": "value"}
        )
