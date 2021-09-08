# Run Book

1. `endgame` script to scrape all the box scores/games
    - Creates `data/{sport}_box.json` and `data/{sport}.csv`
1. `vpp_yoy.ipynb` to translate box scores to VPPs
    - Requires `data/{sport}_box.json` and `data/{sport}.csv`
    - Creates `data/{sport}_values.csv`
1. `python cli.py fit-params data/{sport}_values.csv`
    - Requires `data/{sport}_values.csv` from the previous step
    - Creates `data/{sport}_params.json`
