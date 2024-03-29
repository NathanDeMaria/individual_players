from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np


_DATA_DIR = Path(__file__).parent.parent / "data"


def build_combined_df(gender: str, verbose: bool = True) -> pd.DataFrame:
    player_performances = _load_all_seasons(gender)
    player_performances = _drop_bad_rows(player_performances, verbose)
    player_performances = _drop_bad_players(player_performances, verbose)
    return _add_opponent(player_performances)


def _load_all_seasons(gender: str):
    files = _DATA_DIR.glob(f"all_performances_{gender}_*.csv")
    return pd.concat(pd.read_csv(f) for f in files)


def _drop_bad_rows(player_performances: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Drop any rows that are exactly duplicated or have <=0 possessions"""
    before = len(player_performances)
    player_performances = player_performances.drop_duplicates()
    after = len(player_performances)

    if before > after and verbose:
        print(
            f"Dropped {before - after} duplicate rows. "
            "Maybe you should figure out why they're in here?"
        )

    before = len(player_performances)
    player_performances = player_performances[player_performances.n_possessions > 0]
    after = len(player_performances)

    if before > after and verbose:
        print(
            f"Dropped {before - after} rows because they had <= 0 possessions. "
            "Maybe you should figure out why they're in here?"
        )
    return player_performances


def _drop_bad_players(player_performances: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    player_games = (
        player_performances.groupby(["player_id", "game_id"])
        .agg({"value": "count"})
        .reset_index()
    )

    # Ummmm...this is wrong. What do the duplicates look like?
    has_bad_players = len(player_games[player_games["value"] > 1])

    if not has_bad_players:
        return player_performances
    bad_games = set(player_games[player_games["value"] > 1].game_id)
    if verbose:
        print(
            "Found games where a player had multiple lines in the box score. "
            "Here's an example"
        )
        bad_example = player_games[player_games["value"] > 1].iloc[0]
        print(
            player_performances[
                np.logical_and(
                    player_performances.player_id == bad_example.player_id,
                    player_performances.game_id == bad_example.game_id,
                )
            ]
        )
        print(f"Dropping {len(bad_games)} bad games")
    return player_performances[
        [g not in bad_games for g in player_performances.game_id]
    ]


def _add_opponent(performances: pd.DataFrame) -> pd.DataFrame:
    """Add a column for opp_team_id"""
    unique_games = performances[["game_id", "team_id"]].drop_duplicates()
    game_teams = defaultdict(list)
    for row in unique_games.itertuples():
        game_teams[row.game_id].append(row.team_id)
    assert all(len(teams) == 2 for teams in game_teams.values())

    game_team_df = (
        pd.DataFrame(game_teams)
        .T.reset_index()
        .rename(columns={0: "team_id", 1: "opponent_id", "index": "game_id"})
    )
    other_half = game_team_df.rename(
        columns={"team_id": "opponent_id", "opponent_id": "team_id"}
    )
    game_team_df = pd.concat([game_team_df, other_half], axis=0)

    return performances.merge(game_team_df, on=["game_id", "team_id"])
