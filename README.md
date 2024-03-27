# Run Book

1. `poetry run python create_all_performances.py`
    - Grab the latest performances from S3.
    - This assumes that you've run the `endgame-aws` job
    - This'll pull the data created by that into `data/`
        as files like `all_performances_{league}_{year}.csv`
1. `poetry run python vpp_model.py {league}`
    - Make a model for updating the career VPP given performances.
    - Reads in files like `all_performances_{league}_{year}.csv`
    - Creates `models/{league}.pkl`
1. `defensive_vpp_model.ipynb`
    - Make a model for updating players' defensive VPPs.
    - Reads in:
        - `all_performances_{league}_{year}.csv`
        - `models/{league}.pkl`
    - Creates:
        - `models/{league}_defense.pkl`
1. `adjusted_model.ipynb`
    - Make a model for updating the career VPP given performances adjusted for the defenses they're playing
    - Reads in:
        - `all_performances_{league}_{year}.csv`
        - `models/{league}.pkl`
        - `models/{league}_defense.pkl`
    - Creates:
        - `models/{league}_adjusted.pkl`
1. `poetry run python current_teams.py`
    - Get everybody's ratings given the models created in the previous steps.
    - Reads in:
        - `all_performances_{league}_{year}.csv`
        - `models/{league}.pkl`
        - `models/{league}_defense.pkl`
        - `models/{league}_adjusted.pkl`
    - Creates:
        - `data/{league}_players_ratings.csv`
        - `data/{league}_player_ratings_defense.csv`
1. `poetry run python names.py`
    - Pull extra information on all the players, like names, positions, etc.
    - Reads in:
        - `data/{league}_players_ratings.csv`
    - Creates:
        - `data/{league}_player_info.csv`
1. `team_priors.ipynb`
    - Create a `.csv` of VPP priors for every team based on all of their past players
    - Reads in:
        - `data/{league}_player_ratings.csv`
        - `data/{league}_player_info.csv`
        - `data/{league}_player_ratngs_defense.csv` (I think it's currently unused since this doesn't create defensive priors)
        - `all_performances_{league}_{year}.csv`
    - Creates:
        - `data/{league}_team_priors.csv`
1. `defense_vpp_model_prior.ipynb`
    - Reads in:
        - `models/{league}.pkl`
        - `all_performances_{league}_{year}.csv`
        - `data/{league}_team_priors.csv`
    - Creates:
        - `models/{league}_defense_team_prior.pkl`
1. `adjusted_model_prior.ipynb`
    - Reads in:
        - `all_performances_{league}_{year}.csv`
        - `models/{league}_defense_team_prior.pkl`
    - Creates
        - `models/{league}_adjusted_team_prior.pkl`
1. `poetry run python current_teams.py _team_prior`
    - Creates
        - `data/{league}_player_ratings{prior}.csv`
        - `data/{league}_player_ratings_defense{prior}.csv`
1. `explore.ipynb` to look around at the best this has to offer. General takes:
    - Defense isn't having much effect
        - Partly b/c of that, I didn't implement defensive team priors.
    - The top players are decent, especially womens
    - Top teams are :sus:, I'm not really sure why
        - I thought maybe the PER-based player ratings would be too offensive heavy,
            but it doesn't seem to correlate too much with something I trust.
            Ex: [Massey's offensive ratings](https://masseyratings.com/cbw/ncaa-d1/ratings)
        - Some other gaps:
            - PER-based probably double-counts assisted baskets
            - Recruiting ratings would probably be a better prior
            - The distribution of adjusted player rating isn't normal,
                but I'm treating it like it is
