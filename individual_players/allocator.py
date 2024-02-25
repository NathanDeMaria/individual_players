import pickle
from typing import NamedTuple
import pandas as pd


class MinutesAllocation(NamedTuple):
    possessions_proportion: float
    team_vpp_rank: int


class PossessionAllocator:
    def __init__(self, allocations: list[MinutesAllocation]):
        allocation_df = pd.DataFrame(allocations)
        self._possessions_proportion = (
            allocation_df.groupby("team_vpp_rank")
            .mean()
            .reset_index()
            .possessions_proportion
            # Make sure it's monotonically descending
            .cummin()
        )

    def allocate(self, player_ratings: dict[str, tuple[float, float]]) -> dict[str, float]:
        ranked = sorted(player_ratings.items(), key=lambda kv: kv[1][0], reverse=True)
        scaled_proportions = self._possessions_proportion[: len(ranked)]
        scaled_proportions /= scaled_proportions.sum()
        return {
            player_id: proportion
            for (player_id, _), proportion in zip(ranked, scaled_proportions)
        }

    def save(self, filename: str):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename: str) -> "PossessionAllocator":
        with open(filename, "rb") as file:
            return pickle.load(file)
