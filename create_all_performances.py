import asyncio
from csv import DictWriter
from typing import Iterator
from endgame_aws import read_box_scores, Config

from individual_players.players import PlayerPerformance, get_player_performances
from individual_players.year import get_current_year


async def _read_season_performance(
    bucket: str, league: str, year: int
) -> Iterator[PlayerPerformance]:
    # possessions = await read_possessions(bucket, f"seasons/{year}/{league}.csv")
    # season = (await read_seasons(bucket, f"seasons/{year}/{league}.pkl"))[0]
    season_box_scores = await read_box_scores(
        bucket, f"seasons/{year}/{league}_box.csv"
    )
    # TODO: Figure out why there's nones earlier in the flow
    season_box_scores = [b for b in season_box_scores if b.minutes_played is not None]
    return get_player_performances(season_box_scores)


async def main():
    bucket = Config.init_from_file().bucket
    for league in ["mens", "womens"]:
        for year in range(2022, get_current_year() + 1):
            print(league, year)
            performances = await _read_season_performance(bucket, league, year)
            with open(
                f"data/all_performances_{league}_{year}.csv", "w", encoding="utf-8"
            ) as file:
                writer = DictWriter(
                    file,
                    fieldnames=PlayerPerformance.__dataclass_fields__.keys(),  # pylint: disable=no-member
                )
                writer.writeheader()
                writer.writerows(map(lambda p: p.to_dict(), performances))


if __name__ == "__main__":
    asyncio.run(main())
