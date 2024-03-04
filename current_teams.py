import pandas as pd

from individual_players import (
    build_combined_df,
    callbacks,
    update_loop,
    LeagueModel,
    build_prior_getter,
)
from individual_players.league_model import add_opponent


_LEAGUES = ["womens", "mens"]


def main():
    for league in _LEAGUES:
        performances = build_combined_df(league)
        performances = add_opponent(performances)
        model = LeagueModel.load(f"./models/{league}_league.pkl")
        defense_model = LeagueModel.load(f"./models/{league}_league_defense.pkl")
        adjusted_model = LeagueModel.load(f"./models/{league}_league_adjusted.pkl")
        performances = performances.assign(
            vpp_sd=model.possessions_to_vpp_std(performances.n_possessions),
            defense_sd=defense_model.possessions_to_vpp_std(performances.n_possessions),
            adjusted_vpp_sd=adjusted_model.possessions_to_vpp_std(
                performances.n_possessions
            ),
        )
        defense_callback = callbacks.DefenseAdjustedCallback(
            defense_model, adjusted_model
        )
        _ = update_loop(
            performances,
            model,
            None,
            [defense_callback.team_callback],
            [defense_callback.player_callback],
        )
        league_ratings = pd.DataFrame(
            defense_callback.adjusted_offense_rating
        ).T.rename(columns={0: "vpp", 1: "vpp_var"})
        
        league_ratings.to_csv(f"data/{league}_player_ratings.csv")
        (
            pd.DataFrame(defense_callback.defense_ratings)
            .T.rename(columns={0: "vpp", 1: "vpp_var"})
            .to_csv(f"data/{league}_player_ratings_defense.csv")
        )


if __name__ == "__main__":
    main()
