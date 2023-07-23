from gymnasium.envs.registration import register
from .gymnasium_env_defns import VizdoomScenarioEnv

register(
    id="VizdoomBasic-v0",
    entry_point=VizdoomScenarioEnv,
    kwargs={"level": "basic.cfg"},
)

register(
    id="VizdoomCorridor-v0",
    entry_point=VizdoomScenarioEnv,
    kwargs={"level": "deadly_corridor.cfg"},
)

register(
    id="VizdoomDefendCenter-v0",
    entry_point=VizdoomScenarioEnv,
    kwargs={"level": "defend_the_center.cfg"},
)

register(
    id="VizdoomDefendLine-v0",
    entry_point=VizdoomScenarioEnv,
    kwargs={"level": "defend_the_line.cfg"},
)

register(
    id="VizdoomHealthGathering-v0",
    entry_point=VizdoomScenarioEnv,
    kwargs={"level": "health_gathering.cfg"},
)

register(
    id="VizdoomMyWayHome-v0",
    entry_point=VizdoomScenarioEnv,
    kwargs={"level": "my_way_home.cfg"},
)

register(
    id="VizdoomPredictPosition-v0",
    entry_point=VizdoomScenarioEnv,
    kwargs={"level": "predict_position.cfg"},
)

register(
    id="VizdoomTakeCover-v0",
    entry_point=VizdoomScenarioEnv,
    kwargs={"level": "take_cover.cfg"},
)

register(
    id="VizdoomDeathmatch-v0",
    entry_point=VizdoomScenarioEnv,
    kwargs={"level": "deathmatch.cfg"},
)

register(
    id="VizdoomHealthGatheringSupreme-v0",
    entry_point=VizdoomScenarioEnv,
    kwargs={"level": "health_gathering_supreme.cfg"},
)