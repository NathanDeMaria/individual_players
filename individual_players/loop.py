import pandas as pd
from tqdm import tqdm

from .league_model import LeagueModel
from .callbacks import TeamCallback, PlayerCallback
from .priors import PriorGetter, get_simple_prior, Player
from .ratings import PlayerRatings

def update_loop(
    performances: pd.DataFrame,
    model: LeagueModel,
    prior_getter: PriorGetter = None,
    team_callbacks: list[TeamCallback] = None,
    player_callbacks: list[PlayerCallback] = None,
) -> pd.DataFrame:
    """Assuming the performances DF is sorted in time,
    run the player rating update logic"""
    if prior_getter is None:
        prior_getter = get_simple_prior(model)

    player_ratings = PlayerRatings(prior_getter)
    team_callbacks = team_callbacks or []
    player_callbacks = player_callbacks or []

    for _, game in tqdm(performances.groupby("game_id")):
        for team_id, team in game.groupby("team_id"):
            for team_callback in team_callbacks:
                team_callback(team_id, player_ratings, team)
            for player_performance in team.itertuples():
                for player_callback in player_callbacks:
                    player_callback(team_id, player_performance)
                player = Player(player_performance.player_id, player_performance.team_id)
                current_rating = player_ratings.get_rating(player)
                new_rating = _update_rating(
                    *current_rating, player_performance
                )
                player_ratings.update_rating(player_performance.player_id, new_rating)

    return player_ratings.to_data_frame()


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
