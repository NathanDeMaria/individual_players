import asyncio
from csv import DictWriter
from typing import Iterator
from endgame_aws import read_box_scores, Config

from individual_players.players import PlayerPerformance, get_player_performances


async def _read_season_performance(
    bucket: str, league: str, year: int
) -> Iterator[PlayerPerformance]:
    # possessions = await read_possessions(bucket, f"seasons/{year}/{league}.csv")
    # season = (await read_seasons(bucket, f"seasons/{year}/{league}.pkl"))[0]
    season_box_scores = await read_box_scores(
        bucket, f"seasons/{year}/{league}_box.csv"
    )
    return get_player_performances(season_box_scores)


async def main():
    bucket = Config.init_from_file().bucket
    for league in ["mens", "womens"]:
        for year in range(2010, 2022):
            print(league, year)
            performances = await _read_season_performance(bucket, league, year)
            with open(
                f"all_performances_{league}_{year}.csv", "w", encoding="utf-8"
            ) as file:
                writer = DictWriter(
                    file,
                    fieldnames=PlayerPerformance.__dataclass_fields__.keys(),  # pylint: disable=no-member
                )
                writer.writeheader()
                writer.writerows(map(lambda p: p.to_dict(), performances))


if __name__ == "__main__":
    asyncio.run(main())
