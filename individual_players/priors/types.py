from typing import Callable, NamedTuple


class Player(NamedTuple):
    player_id: str
    team_id: str


class Prior(NamedTuple):
    value: float
    variance: float


PriorGetter = Callable[[Player], Prior]
