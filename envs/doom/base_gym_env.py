import os
import warnings
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
import vizdoom.vizdoom as vzd
from gymnasium.utils import EzPickle


# A fixed set of colors for each potential label
# for rendering an image.
# 256 is not nearly enough for all IDs, but we limit
# ourselves here to avoid hogging too much memory.
LABEL_COLORS = np.random.default_rng(42).uniform(25, 256, size=(256, 3)).astype(np.uint8)


class VizdoomEnv(gym.Env, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": vzd.DEFAULT_TICRATE,
    }

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
        self.frame_skip = frame_skip
        self.depth = depth
        self.labels = labels
        self.automap = automap
        self.game_variables = game_variables
        self.render_mode = render_mode

        # init game
        self.game = vzd.DoomGame()
        if os.path.isabs(level):
            self.config_path = level
        else:
            scenarios_dir = os.path.join(os.path.dirname(__file__), "scenarios")
            self.config_path = os.path.join(scenarios_dir, level)

        self.game.load_config(self.config_path)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.set_window_visible(False)
        self.game.set_depth_buffer_enabled(self.depth)
        self.game.set_labels_buffer_enabled(self.labels)
        self.game.set_automap_buffer_enabled(self.automap)
        self.channels = self.game.get_screen_channels()
        self.state = None
        self.clock = None
        self.window_surface = None
        self.isopen = True

        allowed_buttons = []
        for button in self.game.get_available_buttons():
            if "DELTA" in button.name:
                warnings.warn(
                    f"Removing button {button.name}. DELTA buttons are currently not supported in Gym wrapper. Use binary buttons instead."
                )
            else:
                allowed_buttons.append(button)
        self.game.set_available_buttons(allowed_buttons)
        self.action_space = gym.spaces.Discrete(len(allowed_buttons))

        # specify observation space(s)
        spaces = {
            "screen": gym.spaces.Box(
                0,
                255,
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                    self.channels,
                ),
                dtype=np.uint8,
            )
        }

        if self.depth:
            spaces["depth"] = gym.spaces.Box(
                0,
                255,
                (self.game.get_screen_height(), self.game.get_screen_width(), 1),
                dtype=np.uint8,
            )

        if self.labels:
            spaces["labels"] = gym.spaces.Box(
                0,
                255,
                (self.game.get_screen_height(), self.game.get_screen_width(), 1),
                dtype=np.uint8,
            )

        if self.automap:
            spaces["automap"] = gym.spaces.Box(
                0,
                255,
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                    # "automap" buffer uses same number of channels
                    # as the main screen buffer
                    self.channels,
                ),
                dtype=np.uint8,
            )
        if self.game_variables:
            self.num_game_variables = self.game.get_available_game_variables_size()
            if self.num_game_variables > 0:
                spaces["gamevariables"] = gym.spaces.Box(
                    np.finfo(np.float32).min, np.finfo(np.float32).max, (self.num_game_variables,), dtype=np.float32
                )
        else:
            self.num_game_variables = 0
        if len(spaces) == 1:
            self.observation_space = spaces["screen"]
        else:
            self.observation_space = gym.spaces.Dict(spaces)

        self.game.init()

    def step(self, action):
        # convert action to vizdoom action space (one hot)
        act = np.zeros(self.action_space.n)
        act[action] = 1
        act = np.uint8(act)
        act = act.tolist()

        reward = self.game.make_action(act, self.frame_skip)
        self.state = self.game.get_state()
        done = self.game.is_episode_finished()
        info = {}
        terminated = done
        truncated = False
        if self.render_mode == "human":
            self.render()

        return self.__collect_observations(), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        self.state = self.game.get_state()

        return self.__collect_observations(), {}

    def __collect_observations(self):
        observation = {}
        if self.state is not None:
            if self.channels == 1:
                observation["screen"] = self.state.screen_buffer[..., None]
            else:
                observation["screen"] = np.transpose(self.state.screen_buffer, (1, 2, 0))
            if self.depth:
                observation["depth"] = self.state.depth_buffer[..., None]
            if self.labels:
                observation["labels"] = self.state.labels_buffer[..., None]
            if self.automap:
                if self.channels == 1:
                    observation["automap"] = self.state.automap_buffer[..., None]
                else:
                    observation["automap"] = np.transpose(self.state.automap_buffer, (1, 2, 0))
            if self.num_game_variables > 0:
                observation["gamevariables"] = self.state.game_variables.astype(np.float32)
        else:
            # there is no state in the terminal step, so a zero observation is returned instead
            if isinstance(self.observation_space, gym.spaces.box.Box):
                spaces = dict(screen=self.observation_space)
            else:
                spaces = self.observation_space.spaces
            for space_key, space_item in spaces.items():
                observation[space_key] = np.zeros(space_item.shape, dtype=space_item.dtype)

        # if there is only one observation, return obs as array to sustain compatibility
        if len(observation) == 1:
            observation = observation["screen"]

        return observation

    def __build_human_render_image(self):
        """Stack all available buffers into one for human consumption"""
        game_state = self.game.get_state()
        valid_buffers = game_state is not None

        if not valid_buffers:
            # Return a blank image
            num_enabled_buffers = 1 + self.depth + self.labels + self.automap
            img = np.zeros(
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width() * num_enabled_buffers,
                    3,
                ),
                dtype=np.uint8,
            )
            return img

        if self.channels == 1:
            image_list = [np.repeat(game_state.screen_buffer[..., None], repeats=3, axis=2)]
        else:
            image_list = [np.transpose(game_state.screen_buffer, [1, 2, 0])]

        if self.depth:
            image_list.append(np.repeat(game_state.depth_buffer[..., None], repeats=3, axis=2))

        if self.labels:
            # Give each label a fixed color.
            # We need to connect each pixel in labels_buffer to the corresponding
            # id via `value``
            labels_rgb = np.zeros_like(image_list[0])
            labels_buffer = game_state.labels_buffer
            for label in game_state.labels:
                color = LABEL_COLORS[label.object_id % 256]
                labels_rgb[labels_buffer == label.value] = color
            image_list.append(labels_rgb)

        if self.automap:
            if self.channels == 1:
                image_list.append(np.repeat(game_state.automap_buffer[..., None], repeats=3, axis=2))
            else:
                image_list.append(np.transpose(game_state.automap_buffer, [1, 2, 0]))

        return np.concatenate(image_list, axis=1)

    def render(self):
        if self.clock is None:
            self.clock = pygame.time.Clock()
        render_image = self.__build_human_render_image()
        if self.render_mode == "rgb_array":
            return render_image
        elif self.render_mode == "human":
            # Transpose image (pygame wants (width, height, channels), we have (height, width, channels))
            render_image = render_image.transpose(1, 0, 2)
            if self.window_surface is None:
                pygame.init()
                pygame.display.set_caption("ViZDoom")
                self.window_surface = pygame.display.set_mode(render_image.shape[:2])

            surf = pygame.surfarray.make_surface(render_image)
            self.window_surface.blit(surf, (0, 0))
            pygame.display.update()
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return self.isopen

    def close(self):
        if self.window_surface:
            pygame.quit()
            self.isopen = False
