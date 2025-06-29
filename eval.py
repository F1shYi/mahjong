import torch
from model import CNNModel
from env import MahjongGBEnv
from typing import List, Optional
from feature import FeatureAgent
import numpy as np
from tqdm import tqdm


def get_rewards_one_episode(models: List, env: MahjongGBEnv):

    obs = env.reset()
    done = False
    episode_length = 0
    while not done:
        episode_length += 1
        actions = {}
        values = {}
        for i, agent_name in enumerate(obs):
            model = models[i]
            state = obs[agent_name]
            state['observation'] = torch.tensor(
                state['observation'], dtype=torch.float).unsqueeze(0)
            state['action_mask'] = torch.tensor(
                state['action_mask'], dtype=torch.float).unsqueeze(0)
            model.train(False)  # Batch Norm inference mode
            with torch.no_grad():
                logits, value = model(state)

                action_dist = torch.distributions.Categorical(
                    logits=logits)
                action = action_dist.sample().item()
            actions[agent_name] = action
        next_obs, rewards, done, status = env.step(actions)
        obs = next_obs
    return rewards


def get_rewards(models: List, env: MahjongGBEnv, episodes: int):
    total_rewards = [[], [], [], []]
    for _ in tqdm(range(episodes)):
        rewards = get_rewards_one_episode(models, env)
        rewards_abs_sum = np.sum(np.abs(list(rewards.values())))
        if rewards_abs_sum > 0.0:
            for i, agent_name in enumerate(rewards):
                total_rewards[i].append(rewards[agent_name])
    print(np.mean(total_rewards, axis=1))


if __name__ == "__main__":
    models = [CNNModel(), CNNModel(), CNNModel(), CNNModel()]
    env = MahjongGBEnv(config={'agent_clz': FeatureAgent})
    model_ckpt_paths = [
        r"C:\Courses\S6\MultiAgentSystem\projects\mahjong\output\20250628-233747\model\model_0.pt",
        r"C:\Courses\S6\MultiAgentSystem\projects\mahjong\output\20250628-233747\model\model_10000.pt",
        r"C:\Courses\S6\MultiAgentSystem\projects\mahjong\output\20250628-233747\model\BC_model_160000.pt",
        r"C:\Courses\S6\MultiAgentSystem\projects\mahjong\output\20250628-233747\model\model_20000.pt",
    ]
    for model, ckpt_path in zip(models, model_ckpt_paths):
        if ckpt_path:
            model.load_state_dict(torch.load(ckpt_path))
    get_rewards(models, env, episodes=1000)
