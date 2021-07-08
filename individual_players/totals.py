from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from typing import Counter as CounterType, DefaultDict, Dict, Tuple, List

from endgame.ncaabb.box_score.all import BoxScore

from .constants import POSSESSIONS_PER_FT


@dataclass
class SeasonTotals:
    """
    Totals for a full season for either a league or a team
    """

    field_goal_makes: int
    field_goal_attempts: int
    free_throw_makes: int
    free_throw_attempts: int
    points: int
    turnovers: int
    offensive_rebounds: int
    defensive_rebounds: int
    assists: int

    @property
    def defensive_rebound_pct(self) -> float:
        """
        Pct of rebounds won by the defense
        """
        return self.defensive_rebounds / (
            self.defensive_rebounds + self.offensive_rebounds
        )

    @property
    def free_throw_pct(self) -> float:
        """
        Pct of free throws made
        """
        return self.free_throw_makes / self.free_throw_attempts

    @property
    def assict_pct(self) -> float:
        """
        Pct of made field goals that were assisted
        """
        return self.assists / self.field_goal_makes

    @property
    def factor(self) -> float:
        """
        I haven't thought this all the way through yet, but this is used to
        downweight the value given to made field goals.

        Intuition fuel:
        A league that has a higher assist % will have a lower discount factor here,
        which means made field goals will be valued higher (all else equal).
        """
        return (2 / 3) - (0.5 * self.assists / self.field_goal_makes) / (
            2 * self.field_goal_makes / self.free_throw_makes
        )

    @property
    def value_of_possession(self) -> float:
        """
        How many points a possession is worth, in a pretty straightforward way:
        # of points divided by
        our best estimate for # of possessions
        """
        return self.points / (
            self.field_goal_attempts
            - self.offensive_rebounds
            + self.turnovers
            + POSSESSIONS_PER_FT * self.free_throw_attempts
        )


# This assumes the names of the properties of SeasonTotal
# match the names of PlayerBoxScore
_STATS_TO_TOTAL = frozenset(f.name for f in fields(SeasonTotals))


def compute_season_totals(
    season_games: List[BoxScore],
) -> Tuple[SeasonTotals, Dict[str, SeasonTotals]]:
    """
    league_totals
        Totals of relevant stats for the whole league
    team_totals
        Totals of relevant stats for each team
    """
    totals: CounterType[str] = Counter()
    team_totals: DefaultDict[str, Counter] = defaultdict(Counter)
    for game in season_games:
        for player in game.home.players:
            for stat in _STATS_TO_TOTAL:
                totals[stat] += getattr(player, stat)
                team_totals[game.home.team_id][stat] += getattr(player, stat)
        for player in game.away.players:
            for stat in _STATS_TO_TOTAL:
                totals[stat] += getattr(player, stat)
                team_totals[game.away.team_id][stat] += getattr(player, stat)
    team_total_dict = {
        team_id: SeasonTotals(**team_total)
        for team_id, team_total in team_totals.items()
    }
    return SeasonTotals(**totals), team_total_dict
