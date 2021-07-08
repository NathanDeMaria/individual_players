from endgame.ncaabb.box_score import PlayerBoxScore

from .constants import POSSESSIONS_PER_FT
from .totals import SeasonTotals


def get_unadjusted_rateless_per(
    player_box: PlayerBoxScore, league_stats: SeasonTotals, team_stats: SeasonTotals
) -> float:
    """
    uPER from https://en.wikipedia.org/wiki/Player_efficiency_rating
    before dividing to get per minute #s. This is effectively "total value"
    """
    # Net possessions "won" from things like steals, rebounds, minus turnovers, etc.
    possessions_earned = _calculate_net_possessions_earned(
        player_box, league_stats.defensive_rebound_pct
    )

    return (
        player_box.three_point_makes
        + player_box.assists * 2 / 3
        + player_box.field_goal_makes
        * _calculate_made_fg_value(league_stats, team_stats)
        + player_box.free_throw_makes * _calculate_made_ft_value(team_stats)
        + league_stats.value_of_possession * possessions_earned
        - player_box.fouls * _calculate_foul_value_lost(league_stats)
    )


def _calculate_made_fg_value(
    league_stats: SeasonTotals, team_stats: SeasonTotals
) -> float:
    return 2 - league_stats.factor * team_stats.assict_pct


def _calculate_made_ft_value(team_stats: SeasonTotals) -> float:
    return 0.5 * (2 - team_stats.assict_pct / 3)


def _calculate_net_possessions_earned(
    player_box: PlayerBoxScore, defensive_rebound_percentage: float
) -> float:
    net_possessions = 0

    net_possessions += defensive_rebound_percentage * player_box.offensive_rebounds
    net_possessions += player_box.steals
    net_possessions += defensive_rebound_percentage * player_box.blocks

    net_possessions -= player_box.turnovers
    fg_misses = player_box.field_goal_attempts - player_box.field_goal_makes
    net_possessions -= defensive_rebound_percentage * fg_misses

    free_throw_misses = player_box.free_throw_attempts - player_box.free_throw_makes
    possessions_from_missed_fts = (
        POSSESSIONS_PER_FT
        * (POSSESSIONS_PER_FT + (1 - POSSESSIONS_PER_FT) * defensive_rebound_percentage)
        * free_throw_misses
    )
    net_possessions -= possessions_from_missed_fts

    return net_possessions


def _calculate_foul_value_lost(league_stats: SeasonTotals) -> float:
    return (
        league_stats.free_throw_pct
        - POSSESSIONS_PER_FT
        * league_stats.free_throw_pct
        * league_stats.value_of_possession
    )
