from typing import Callable
import pandas as pd

from ..types import PlayerRatings

TeamCallback = Callable[[str, PlayerRatings, pd.DataFrame], None]
PlayerCallback = Callable[[str, pd.Series], None]
