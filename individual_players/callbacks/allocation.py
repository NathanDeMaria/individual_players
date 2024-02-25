import pandas as pd

from ..allocator import MinutesAllocation, PossessionAllocator
from .types import TeamCallback, PlayerCallback
from ..ratings import PlayerRatings, Player


class PossessionAllocationCallbacks:
    def __init__(self):
        self._team_ranks: dict[str, dict[str, int]] = {}
        self._allocations: list[MinutesAllocation] = []

    @property
    def team_callback(self) -> TeamCallback:
        def set_pregame_ratings(
            team_id: str, player_ratings: PlayerRatings, team: pd.DataFrame
        ):
            pregame_ratings = {
                player_performance.player_id: player_ratings.get_rating(Player(player_performance.player_id, team_id))[0]
                for player_performance in team.itertuples()
            }
            self._team_ranks[team_id] = {
                pid: i + 1
                for i, (pid, _) in enumerate(
                    sorted(pregame_ratings.items(), key=lambda kv: kv[1], reverse=True)
                )
            }

        return set_pregame_ratings

    @property
    def player_callback(self) -> PlayerCallback:
        def add_allocation(team_id: str, player_performance):
            team_rank = self._team_ranks[team_id][player_performance.player_id]
            self._allocations.append(
                MinutesAllocation(
                    possessions_proportion=player_performance.possessions_proportion,
                    team_vpp_rank=team_rank,
                )
            )

        return add_allocation

    @property
    def allocator(self) -> PossessionAllocator:
        return PossessionAllocator(self._allocations)
