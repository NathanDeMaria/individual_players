import pandas as pd

from .priors import Player, PriorGetter


_Rating = tuple[float, float]


class PlayerRatings:
    def __init__(self, prior_getter: PriorGetter) -> None:
        self._ratings: dict[str, _Rating] = {}
        self._prior_getter = prior_getter

    def get_rating(self, player: Player) -> _Rating:
        rating = self._ratings.get(player.player_id)
        if rating is not None:
            return rating
        return self._prior_getter(player)

    def update_rating(self, player_id: str, new_rating: _Rating) -> None:
        self._ratings[player_id] = new_rating

    def to_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._ratings).T.rename(columns={0: "vpp", 1: "vpp_var"})
