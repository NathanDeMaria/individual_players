# mypy: disable-error-code="arg-type"
# because there's lots of pandas in here
import pandas as pd

from .types import PriorGetter, Prior, Player


def build_prior_getter(league: str) -> PriorGetter:
    """Prior based on your team"""
    team_priors = pd.read_csv(f"./data/{league}_team_priors.csv", index_col=0)

    default_prior = Prior(team_priors.vpp.median(), team_priors.vpp_var.max())

    def _get_prior(player: Player) -> Prior:
        # TODO: default if the team's not in there yet
        try:
            team_prior = team_priors.loc[player.team_id]
        except KeyError:
            return default_prior
        return Prior(team_prior.vpp, team_prior.vpp_var)

    return _get_prior
