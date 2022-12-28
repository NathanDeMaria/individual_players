from fire import Fire

from individual_players import LeagueModel, build_combined_df
from individual_players.league_model import (
    add_player_aggregates,
    fit_std_by_sample_size,
)


def main(gender: str):
    """Script version of vpp_model.ipynb

    Creates a saved model file that'll have all the info it needs to create
    priors of player performances (measured by VPP) and update them.
    """
    performances = build_combined_df(gender)
    with_player_aggregates, by_player = add_player_aggregates(performances)
    _, get_vpp_sd = fit_std_by_sample_size(with_player_aggregates)
    career_vpp = by_player.total_value / by_player.total_possessions
    model = LeagueModel(
        possessions_to_vpp_std=get_vpp_sd,
        vpp_mean=career_vpp.mean(),
        vpp_variance=career_vpp.var(),
    )
    model.save(f"models/{gender}_league.pkl")


if __name__ == "__main__":
    Fire(main)
