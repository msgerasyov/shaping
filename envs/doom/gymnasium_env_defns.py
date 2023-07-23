from typing import Optional

from .base_gym_env import VizdoomEnv
from gymnasium.utils import EzPickle


class VizdoomScenarioEnv(VizdoomEnv, EzPickle):
    """Basic ViZDoom environments which reside in the `scenarios` directory"""

    def __init__(
        self,
        level: str,
        frame_skip: int = 1,
        depth: bool = False,
        labels: bool = False,
        automap: bool = False,
        game_variables: bool = False,
        render_mode: Optional[str] = None,
    ):
        EzPickle.__init__(self, level, frame_skip, depth, labels, automap, game_variables, render_mode)
        super().__init__(level, frame_skip, depth, labels, automap, game_variables, render_mode)
