from pathlib import Path
from fire import Fire
from individual_players import LeagueModel, build_combined_df, update_loop, callbacks


def main(league: str):
    performances = build_combined_df(league, verbose=False)
    model = LeagueModel.load(str(Path("models", f"{league}_league.pkl")))
    performances = performances.assign(
        vpp_sd=model.possessions_to_vpp_std(performances.n_possessions)
    )
    # TODO: actually sort by date
    performances = performances.sort_values("game_id")
    game_possessions = (
        performances.groupby("game_id")
        .agg({"n_possessions": "sum"})
        .reset_index()
        .assign(game_possessions=lambda _: _.n_possessions / 5 / 2)[
            ["game_possessions", "game_id"]
        ]
    )
    performances = performances.merge(game_possessions, on="game_id").assign(
        possessions_proportion=lambda _: _.n_possessions / (_.game_possessions * 5)
    )
    allocations = callbacks.PossessionAllocationCallbacks()
    update_loop(
        performances,
        model,
        team_callbacks=[allocations.team_callback],
        player_callbacks=[allocations.player_callback],
    )
    allocations.allocator.save(str(Path("models", f"{league}_allocator.pkl")))


if __name__ == "__main__":
    Fire(main)
