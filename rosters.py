import json
from individual_players import build_combined_df, update_loop, LeagueModel


_LEAGUES = ["womens", "mens"]


def _save_rosters() -> None:
    """Build rosters.json that links players/teams/ratings."""
    for league in _LEAGUES:
        performances = build_combined_df(league)
        model = LeagueModel.load(f"./models/{league}_league.pkl")
        performances = performances.assign(
            vpp_sd=model.possessions_to_vpp_std(performances.n_possessions),
        )
        league_ratings = update_loop(
            performances,
            model,
            None,
            [],
            [],
        )
        rosters = {
            team_id: team[team.game_id == team.game_id.max()]
            .merge(league_ratings, left_on="player_id", right_index=True)
            .player_id.tolist()
            for team_id, team in performances.groupby("team_id")
        }
        with open(f"data/{league}_rosters.json", "w", encoding="utf-8") as file:
            json.dump(rosters, file)


if __name__ == "__main__":
    _save_rosters()
