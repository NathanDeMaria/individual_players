from ..league_model import LeagueModel
from .types import PriorGetter, Prior


def get_simple_prior(model: LeagueModel) -> PriorGetter:
    """Simple prior, just the same for everybody"""
    return lambda _: Prior(model.vpp_mean, model.vpp_variance)
