from endgame.ncaabb.box_score import PlayerBoxScore


class FilledOutBoxScore(PlayerBoxScore):
    minutes_played: int
    field_goal_makes: int
    field_goal_attempts: int
    three_point_makes: int
    three_point_attempts: int
    free_throw_makes: int
    free_throw_attempts: int
    offensive_rebounds: int
    defensive_rebounds: int
    rebounds: int
    assists: int
    steals: int
    blocks: int
    turnovers: int
    fouls: int
    points: int

    @classmethod
    def from_box_score(cls, box_score: PlayerBoxScore):
        return cls.from_dict(box_score.to_dict())
