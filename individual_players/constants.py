# Used to estimate out the # of possessions implied by each free throw
# (without knowing which were and-1's, 2 shots, 1-and-1's, 3 shots)
# Hollinger's PER uses 0.44 for the NBA,
# but KenPom uses this value https://kenpom.com/blog/stats-explained/
# We could compute this per league if we use play-by-play data
POSSESSIONS_PER_FT = 0.475
