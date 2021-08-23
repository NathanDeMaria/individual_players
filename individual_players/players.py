from logging import getLogger
from typing import List

from endgame.ncaabb import PlayerBoxScore, BoxScore

from .constants import POSSESSIONS_PER_FT
from .per import get_unadjusted_rateless_per
from .totals import compute_season_totals


logger = getLogger(__name__)


def get_player_performances(game_box_scores: List[BoxScore]):
    """Get all the player performances for this season"""
    totals, team_totals = compute_season_totals(game_box_scores)

    if totals.assict_pct <= 0.1:
        logger.info("Skipping season b/c it doesn't seem like assists were counted")
        return []
    if totals.turnovers / totals.assists < 0.1:
        logger.info("Skipping season b/c it doesn't seem like turnovers were counted")
        return []

    values = []
    for game_box_score in game_box_scores:
        game_possessions = _get_possessions(game_box_score)
        if game_possessions == 0:
            continue
        player_possessions = distribute_player_possessions(
            game_box_score.home.players, game_possessions / 2
        )
        for player, possessions in zip(game_box_score.home.players, player_possessions):
            value = get_unadjusted_rateless_per(
                player, totals, team_totals[game_box_score.home.team_id]
            )
            values.append(
                (
                    player.player_id,
                    value,
                    game_box_score.game_id,
                    game_box_score.home.team_id,
                    possessions,
                )
            )
        player_possessions = distribute_player_possessions(
            game_box_score.away.players, game_possessions / 2
        )
        for player, possessions in zip(game_box_score.away.players, player_possessions):
            value = get_unadjusted_rateless_per(
                player, totals, team_totals[game_box_score.away.team_id]
            )
            values.append(
                (
                    player.player_id,
                    value,
                    game_box_score.game_id,
                    game_box_score.away.team_id,
                    possessions,
                )
            )
    return values


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


def _get_possessions(game_box_score: BoxScore) -> float:
    return sum(_get_player_possessions(p) for p in game_box_score.home.players) + sum(
        _get_player_possessions(p) for p in game_box_score.away.players
    )
