from collections import defaultdict
import pandas as pd
from tqdm import tqdm

from .league_model import LeagueModel
from .callbacks import TeamCallback, PlayerCallback


def update_loop(
    performances: pd.DataFrame,
    model: LeagueModel,
    team_callbacks: list[TeamCallback] = None,
    player_callbacks: list[PlayerCallback] = None,
) -> pd.DataFrame:
    """Assuming the performances DF is sorted in time,
    run the player rating update logic"""
    player_ratings: defaultdict[str, tuple[float, float]] = defaultdict(
        lambda: (model.vpp_mean, model.vpp_variance)
    )
    team_callbacks = team_callbacks or []
    player_callbacks = player_callbacks or []

    for _, game in tqdm(performances.groupby("game_id")):
        for team_id, team in game.groupby("team_id"):
            for team_callback in team_callbacks:
                team_callback(team_id, player_ratings, team)
            for player_performance in team.itertuples():
                for player_callback in player_callbacks:
                    player_callback(team_id, player_performance)
                player_ratings[player_performance.player_id] = _update_rating(
                    *player_ratings[player_performance.player_id], player_performance
                )

    return pd.DataFrame(player_ratings).T.rename(columns={0: "vpp", 1: "vpp_var"})


def _update_rating(
    player_mu: float, player_var: float, player_performance
) -> tuple[float, float]:
    # Rating Update
    game_vpp = player_performance.value / player_performance.n_possessions
    new_mu = (player_mu * (player_performance.vpp_sd**2) + game_vpp * player_var) / (
        (player_performance.vpp_sd**2) + player_var
    )
    new_var = 1 / ((1 / player_var + 1 / (player_performance.vpp_sd**2)))
    return new_mu, new_var
