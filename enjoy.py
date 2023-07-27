import argparse
import glob
import os
from distutils.util import strtobool
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

import envs
import wandb
from main import Agent, make_env


def evaluate(
    agent_path: str,
    make_env: Callable,
    env_id: str,
    run_name: str,
    seed: int = 0,
    eval_episodes: int = 10,
    verbose: bool = True,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, seed, 0, capture_video, run_name)])
    agent = torch.load(agent_path).to(device)
    agent.eval()

    episodic_returns = []
    for i in range(eval_episodes):
        obs, _ = envs.reset()
        done = False
        while not done:
            obs = torch.Tensor(obs).to(device)
            action, _, _, _ = agent.get_action_and_value(obs)
            obs, _, _, _, infos = envs.step(action.cpu().numpy())
            if "final_info" in infos.keys() and "episode" in infos["final_info"][0].keys():
                episodic_returns.append(infos["final_info"][0]["episode"]["r"][0])
                done = True
        if verbose:
            print(f"episode={i}, return={episodic_returns[-1]}")
    envs.close()

    return episodic_returns


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-path", type=str, default="",
        help="path to the saved agent (either local or wandb)")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
    parser.add_argument("--n-eval-episodes", type=int, default=10,
        help="the number of eval episodes")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-run-path", type=str, default="",
        help="the wandb's run path")
    parser.add_argument("--wandb-download", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to dowload the agent from Weights and Biases")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.wandb_download:
        wandb.restore(args.agent_path, run_path=args.wandb_run_path)
    run_name = os.path.split(os.path.dirname(args.agent_path))[-1]
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    episodic_returns = evaluate(
        args.agent_path,
        make_env,
        args.env_id,
        run_name,
        args.seed,
        args.n_eval_episodes,
        device=device,
        capture_video=args.capture_video,
    )
    print(f"mean_episodic_return={np.mean(episodic_returns):.3f}+/-{np.std(episodic_returns):.3f}")
    if args.track:
        run = wandb.init(id=wandb.Api().run(args.wandb_run_path).id, resume='allow')
        if args.capture_video:
            for fname in glob.glob(os.path.join("videos", run_name, "*.mp4")):
                run.log({"videos": wandb.Video(fname, fps=30)})
        run.summary["eval/episodic_return_mean"] = np.mean(episodic_returns)
        run.summary["eval/episodic_return_std"] = np.std(episodic_returns)
