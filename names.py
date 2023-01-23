import asyncio
import re
from functools import partial
from pathlib import Path
from typing import Optional
import pandas as pd
from bs4 import BeautifulSoup
from fire import Fire

from endgame.web import get


def _build_link(league: str, player_id: str) -> str:
    return f"https://www.espn.com/{league}-college-basketball/player/_/id/{player_id}"


def _get_team_link(league: str, raw, team_info) -> Optional[str]:
    if match := re.search(
        f"https://www.espn.com/{league}-college-basketball/team/_/id/[0-9]+/[0-9a-z\\-]+",
        raw,
    ):
        return match.group()
    if team_link := team_info.select_one("a"):
        return team_link.attrs["href"]
    return None


async def _get_player_info(league: str, player_id: str) -> dict:
    url = _build_link(league, player_id)
    response = await get(url)
    soup = BeautifulSoup(response.data, features="html.parser")

    player_name = " ".join(
        span.text for span in soup.select("h1.PlayerHeader__Name span")
    )

    team_info = soup.select_one("ul.PlayerHeader__Team_Info")
    pieces = team_info.select("li")
    if len(pieces) > 1:
        *_, number, position = pieces
        position = position.text.strip()
        number = number.text.strip()
    else:
        position = pieces[0].text.strip()
        number = "?"

    if team_link := _get_team_link(league, response.data.decode(), team_info):
        team_id, team_name = team_link.split("/")[-2:]
    else:
        # TODO: find other ways...I think there's other links to the team page in here
        team_id = "?"
        team_name = "?"

    other_columns = {
        item.select_one("div.ttu").text: item.select_one("div.fw-medium").text
        for item in soup.select("ul.PlayerHeader__Bio_List li")
    }
    await response.save_if_necessary()

    return {
        "player_id": player_id,
        "team_id": team_id,
        "team_name": team_name,
        "number": number,
        "position": position,
        "player_name": player_name,
        **other_columns,
    }


async def _try_get_player_info(league: str, player_id: str) -> Optional[dict]:
    try:
        return await _get_player_info(league, player_id)
    except Exception as error:
        print(f"Problem with {_build_link(league, player_id)} {str(error)}")
        return None


_LEAGUES = ["mens", "womens"]


def _get_all_player_ids(league: str) -> list[int]:
    player_ratings = pd.read_csv(f"./data/{league}_player_ratings.csv", index_col=0)
    return player_ratings.index


def _get_starting_point(csv_path: Path) -> tuple[pd.DataFrame, set[int]]:
    if not csv_path.exists():
        return pd.DataFrame(), set()
    starting = pd.read_csv(csv_path)
    return starting, {int(pid) for pid in starting.player_id}


async def main():
    """Get a .csv that has some data on every player.

    This is meant to be re-run occasionally, and will only get the web page
    for new players/players it previously failed on.

    There's probably more info you could parse in here,
    and definitely some edge cases that can be handled.
    I figure this is a fine start, and if important players end up with ?'s,
    then I can do something about it
    """
    for league in _LEAGUES:
        player_ids = _get_all_player_ids(league)
        output_path = Path("data", f"{league}_player_info.csv")
        so_far, done_ids = _get_starting_point(output_path)

        to_get = filter(lambda pid: pid not in done_ids, player_ids)
        tasks = map(partial(_try_get_player_info, league), to_get)
        player_infos = await asyncio.gather(*tasks)
        added = pd.DataFrame([i for i in player_infos if i is not None])
        print(f"Found {len(added)} more players")

        pd.concat([so_far, added]).to_csv(output_path)


if __name__ == "__main__":
    Fire(main)
