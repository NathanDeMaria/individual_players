import json
import pandas as pd

from individual_players import (
    build_combined_df,
    callbacks,
    update_loop,
    LeagueModel,
)
from individual_players.league_model import add_opponent


_LEAGUES = ["womens", "mens"]


def main():
    for league in _LEAGUES:
        performances = build_combined_df(league)
        performances = add_opponent(performances)
        model = LeagueModel.load(f"./models/{league}_league.pkl")
        defense_model = LeagueModel.load(f"./models/{league}_league_defense.pkl")
        performances = performances.assign(
            vpp_sd=model.possessions_to_vpp_std(performances.n_possessions),
            defense_sd=defense_model.possessions_to_vpp_std(performances.n_possessions),
        )
        defense_callback = callbacks.DefenseAdjustedCallback(defense_model, model)
        _ = update_loop(
            performances,
            model,
            [defense_callback.team_callback],
            [defense_callback.player_callback],
        )
        league_ratings = pd.DataFrame(
            defense_callback.adjusted_offense_rating
        ).T.rename(columns={0: "vpp", 1: "vpp_var"})
        rosters = {
            team_id: team[team.game_id == team.game_id.max()]
            .merge(league_ratings, left_on="player_id", right_index=True)
            .player_id.tolist()
            for team_id, team in performances.groupby("team_id")
        }
        with open(f"data/{league}_rosters.json", "w", encoding="utf-8") as file:
            json.dump(rosters, file)
        league_ratings.to_csv(f"data/{league}_player_ratings.csv")


if __name__ == "__main__":
    main()
