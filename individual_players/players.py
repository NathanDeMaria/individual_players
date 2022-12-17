from dataclasses import dataclass
from itertools import groupby
from logging import getLogger
from typing import Callable, Iterator, List
from dataclasses_json import DataClassJsonMixin
from endgame.ncaabb import PlayerBoxScore
from endgame_aws import FlattenedBoxScore

from .constants import POSSESSIONS_PER_FT
from .per import get_unadjusted_rateless_per
from .totals import SeasonTotals, compute_season_totals


logger = getLogger(__name__)


@dataclass
class PlayerPerformance(DataClassJsonMixin):
    player_id: str
    value: float
    game_id: str
    team_id: str
    # This is a float b/c it's an estimate
    n_possessions: float


_GAME_KEY: Callable[[FlattenedBoxScore], str] = lambda s: s.game_id
_TEAM_KEY: Callable[[FlattenedBoxScore], str] = lambda s: s.team_id


def get_player_performances(
    season_box_scores: List[FlattenedBoxScore],
) -> Iterator[PlayerPerformance]:
    """Get all the player performances for this season"""
    totals, team_totals = compute_season_totals(season_box_scores)

    if totals.assict_pct <= 0.1:
        logger.info("Skipping season b/c it doesn't seem like assists were counted")
        return
    if totals.turnovers / totals.assists < 0.1:
        logger.info("Skipping season b/c it doesn't seem like turnovers were counted")
        return

    for game_id, game_box_scores in groupby(
        sorted(season_box_scores, key=_GAME_KEY), key=_GAME_KEY
    ):
        yield from _get_game_performances(
            list(game_box_scores), game_id, totals, team_totals
        )


def _get_game_performances(
    game_box_scores: list[FlattenedBoxScore],
    game_id: str,
    totals: SeasonTotals,
    team_totals: dict[str, SeasonTotals],
) -> Iterator[PlayerPerformance]:
    game_possessions = _get_possessions(game_box_scores)
    if game_possessions == 0:
        return
    for team_id, team_players in groupby(
        sorted(game_box_scores, key=_TEAM_KEY), key=_TEAM_KEY
    ):
        player_list = list(team_players)
        player_possessions = distribute_player_possessions(
            player_list, game_possessions / 2
        )
        for player, possessions in zip(player_list, player_possessions):
            value = get_unadjusted_rateless_per(player, totals, team_totals[team_id])
            yield PlayerPerformance(
                player_id=player.player_id,
                value=value,
                game_id=game_id,
                team_id=team_id,
                n_possessions=possessions,
            )


def distribute_player_possessions(
    players: List[PlayerBoxScore], team_total_possessions: float
) -> List[float]:
    """
    Get the # of possessions each player was on the court for

    Fallback order:
    - Get the raw # of possessions each player was on the court from play-by-play data
        - Not implemented (I just don't have that data yet)
    - Divide the total possessions by the % of minutes played by the player
        - Assumes constant pace with/without the player, that's fine
    - Find the # of possessions a player used themselves, use that to divvy them up
        - Not great, but lots of games don't have minutes data in the box score
    """
    # If there's 5 or fewer players, they were all on the court for the whole game
    if len(players) <= 5:
        return [team_total_possessions for _ in players]
    if any(p.minutes_played is not None and p.minutes_played != 0 for p in players):
        return _divvy_possessions_by_minutes(
            [p.minutes_played for p in players], team_total_possessions
        )
    return _divvy_possessions_by_usage(
        [_get_player_possessions(p) for p in players], team_total_possessions
    )


def _divvy_possessions_by_minutes(
    player_minutes: List[float], team_total_possessions: float
) -> List[float]:
    total_minutes = sum(player_minutes)
    return [5 * team_total_possessions * m / total_minutes for m in player_minutes]


def _divvy_possessions_by_usage(
    player_possessions: List[float], team_total_possessions: float
) -> List[float]:
    # If any players had more than 20% of the possessions, limit them to 20%
    max_possessions = team_total_possessions * 0.2
    total_below_one_fifth = sum(p for p in player_possessions if p < max_possessions)
    n_above_one_fifth = sum(p >= max_possessions for p in player_possessions)
    non_max_player_possessions = (
        team_total_possessions - n_above_one_fifth * max_possessions
    )

    return [
        5 * min(max_possessions, p / total_below_one_fifth * non_max_player_possessions)
        for p in player_possessions
    ]


# TODO: put these on the box score classes? Or some static utils module?
def _get_player_possessions(player_box_score: PlayerBoxScore) -> float:
    return (
        player_box_score.field_goal_attempts
        - player_box_score.offensive_rebounds
        + player_box_score.turnovers
        + POSSESSIONS_PER_FT * player_box_score.free_throw_attempts
    )


def _get_possessions(player_box_scores: list[FlattenedBoxScore]) -> float:
    return sum(map(_get_player_possessions, player_box_scores))
