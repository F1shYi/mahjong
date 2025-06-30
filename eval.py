import queue
import time
import multiprocessing as mp
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


def worker(progress_queue, result_queue, models, env, episodes_per_process):
    local_rewards = [[], [], [], []]
    for _ in range(episodes_per_process):
        # 这里是你的评估逻辑
        rewards = get_rewards_one_episode(models, env)
        rewards_abs_sum = np.sum(np.abs(list(rewards.values())))
        if rewards_abs_sum > 0.0:
            for i, agent_name in enumerate(rewards):
                local_rewards[i].append(rewards[agent_name])
        progress_queue.put(1)  # 通知主进程完成了一个episode
    result_queue.put(local_rewards)


def get_rewards_parallel(models, env, episodes, num_processes=4):
    progress_queue = mp.Queue()
    result_queue = mp.Queue()
    episodes_per_process = episodes // num_processes
    processes = []

    pbar = tqdm(total=episodes, desc="Total episodes")
    pbar.refresh()

    for _ in range(num_processes):
        p = mp.Process(target=worker, args=(
            progress_queue, result_queue, models, env, episodes_per_process))
        p.start()
        processes.append(p)

    total_rewards = [[], [], [], []]

    finished_processes = 0
    while finished_processes < num_processes:
        try:
            while True:
                progress_queue.get_nowait()
                pbar.update(1)
        except queue.Empty:
            pass

        finished_processes = sum(1 for p in processes if not p.is_alive())
        time.sleep(0.1)
    # 收集所有结果
    for _ in range(num_processes):
        local_rewards = result_queue.get()
        for i in range(4):
            total_rewards[i].extend(local_rewards[i])

    for p in processes:
        p.join()

    pbar.close()
    print(np.mean(total_rewards, axis=1))


if __name__ == "__main__":
    models = [CNNModel(), CNNModel(), CNNModel(), CNNModel()]
    env = MahjongGBEnv(config={'agent_clz': FeatureAgent})
    model_ckpt_paths = [
        None,
        r"C:\Courses\S6\MultiAgentSystem\projects\mahjong\output\20250628-233747\model\BC_model_80000.pt",
        r"C:\Courses\S6\MultiAgentSystem\projects\mahjong\output\20250628-233747\model\BC_model_160000.pt",
        r"C:\Courses\S6\MultiAgentSystem\projects\mahjong\output\20250628-233747\model\BC_model_240000.pt",
    ]
    for model, ckpt_path in zip(models, model_ckpt_paths):
        if ckpt_path:
            model.load_state_dict(torch.load(ckpt_path))

    num_processes = mp.cpu_count()  # 使用所有可用的CPU核心
    get_rewards_parallel(models, env, episodes=2000,
                         num_processes=num_processes)
