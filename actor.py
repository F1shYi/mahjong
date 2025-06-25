from multiprocessing import Process
import numpy as np
import torch

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import MahjongGBEnv
from feature import FeatureAgent
from model import CNNModel
from config import Config
from learner import Learner
from typing import List, Optional
import time


class Actor(Process):
    def __init__(self,
                 config: Config,
                 replay_buffer: ReplayBuffer,
                 actor_id: int,
                 learner: Learner,
                 full_shanten_tilewalls: Optional[List[List[str]]] = None):
        super(Actor, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.actor_id = actor_id
        self.learner = learner
        self.prev_training_stage = None  # 用于检测阶段切换

        # curriculum 数据切片
        if self.config.use_curriculum and full_shanten_tilewalls:
            self.shanten_tilewalls: List[List[str]] = []
            for shanten_tilewalls in full_shanten_tilewalls:
                l = len(shanten_tilewalls)
                per_actor = l // self.config.num_actors
                start_idx = self.actor_id * per_actor
                end_idx = (self.actor_id + 1) * \
                    per_actor if self.actor_id < self.config.num_actors - 1 else l
                self.shanten_tilewalls.append(
                    shanten_tilewalls[start_idx:end_idx])
            self.current_shanten = 0
            self.shanten_tilewalls_idx = 0
            for shanten_num, shanten_tilewalls in enumerate(self.shanten_tilewalls):
                print(
                    f"Actor {self.actor_id} using {shanten_num}-shanten data, len={len(shanten_tilewalls)}")
        else:
            self.shanten_tilewalls = []
            self.current_shanten = 0
            self.shanten_tilewalls_idx = 0

    def maybe_reset_curriculum(self):
        """检测 BC 和 PPO 阶段切换，并重置 shanten 状态"""
        current_stage = 'bc' if self.learner.get_BC_ongoing() else 'ppo'

        if self.prev_training_stage is not None and self.prev_training_stage != current_stage:
            print(
                f"[Actor {self.actor_id}] Training stage switched from {self.prev_training_stage} to {current_stage}. Reset curriculum state.")
            self.current_shanten = 0
            self.shanten_tilewalls_idx = 0

        self.prev_training_stage = current_stage

    def get_tilewalls(self) -> str:
        is_bc = self.learner.get_BC_ongoing()
        current_iteration = self.learner.get_iteration()

        # 不使用 curriculum，直接返回空环境
        if (not is_bc and not self.config.use_curriculum) or (is_bc and not self.config.use_curriculum_for_BC):
            return ''

        curriculum_iters = self.config.curriculum_iterations_for_BC if is_bc else self.config.curriculum_iterations

        if self.current_shanten >= len(curriculum_iters):
            return ''

        while self.current_shanten < len(curriculum_iters) and current_iteration >= curriculum_iters[self.current_shanten]:
            print(f"[Actor {self.actor_id}] Iteration {current_iteration} reached threshold {curriculum_iters[self.current_shanten]}, switching shanten {self.current_shanten} -> {self.current_shanten + 1}")
            self.current_shanten += 1
            self.shanten_tilewalls_idx = 0

        if self.current_shanten >= len(self.shanten_tilewalls):
            return ''

        tilewall = self.shanten_tilewalls[self.current_shanten][self.shanten_tilewalls_idx]
        self.shanten_tilewalls_idx = (
            self.shanten_tilewalls_idx + 1) % len(self.shanten_tilewalls[self.current_shanten])
        return tilewall

    def run(self):
        torch.set_num_threads(1)
        model_pool = ModelPoolClient(self.config.model_pool_name)
        model = CNNModel()

        version = model_pool.get_latest_model()
        model.load_state_dict(model_pool.load_model(version))

        env = MahjongGBEnv(config={'agent_clz': FeatureAgent})
        policies = {player: model for player in env.agent_names}

        self.prev_training_stage = None  # 用于检测阶段切换

        while True:

            # 退出条件判断
            is_bc_stage = self.learner.get_BC_ongoing()
            iteration = self.learner.get_iteration()
            if (not is_bc_stage) and (iteration >= self.config.max_iterations):
                print(
                    f"[Actor {self.actor_id}] PPO stage and iteration {iteration} > max_iterations. Exiting.")
                break

            # 检测是否切换训练阶段，重置curriculum
            self.maybe_reset_curriculum()

            # 更新模型
            latest = model_pool.get_latest_model()
            if latest['id'] > version['id']:
                model.load_state_dict(model_pool.load_model(latest))
                version = latest

            # 环境重置，传入当前tilewall
            obs = env.reset(tileWall=self.get_tilewalls())
            episode_data = {agent_name: {
                'state': {'observation': [], 'action_mask': []},
                'action': [], 'reward': [], 'value': []
            } for agent_name in env.agent_names}
            done = False
            while not done:
                actions, values = {}, {}
                for agent_name in obs:
                    agent_data = episode_data[agent_name]
                    state = obs[agent_name]
                    agent_data['state']['observation'].append(
                        state['observation'])
                    agent_data['state']['action_mask'].append(
                        state['action_mask'])

                    state['observation'] = torch.tensor(
                        state['observation'], dtype=torch.float).unsqueeze(0)
                    state['action_mask'] = torch.tensor(
                        state['action_mask'], dtype=torch.float).unsqueeze(0)

                    model.eval()
                    with torch.no_grad():
                        logits, value = model(state)
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample().item()
                        value = value.item()

                    actions[agent_name] = action
                    values[agent_name] = value
                    agent_data['action'].append(action)
                    agent_data['value'].append(value)

                next_obs, rewards, done, status = env.step(actions)
                for agent_name in rewards:
                    episode_data[agent_name]['reward'].append(
                        rewards[agent_name])
                obs = next_obs

            # 数据处理和存入 replay buffer
            if self.learner.get_BC_ongoing():
                for agent_name, reward in rewards.items():
                    if reward <= 0:
                        continue
                    agent_data = episode_data[agent_name]
                    try:
                        if len(agent_data['action']) < len(agent_data['reward']):
                            agent_data['reward'].pop(0)
                        obs = np.stack(agent_data['state']['observation'])
                        mask = np.stack(agent_data['state']['action_mask'])
                        actions = np.array(
                            agent_data['action'], dtype=np.int64)

                        self.replay_buffer.push({
                            'state': {'observation': obs, 'action_mask': mask},
                            'action': actions,
                            'adv': np.zeros_like(actions, dtype=np.float32),
                            'target': np.zeros_like(actions, dtype=np.float32)
                        })
                    except:
                        continue
            else:
                for agent_name, agent_data in episode_data.items():
                    try:
                        if len(agent_data['action']) < len(agent_data['reward']):
                            agent_data['reward'].pop(0)
                        obs = np.stack(agent_data['state']['observation'])
                        mask = np.stack(agent_data['state']['action_mask'])
                        actions = np.array(
                            agent_data['action'], dtype=np.int64)
                        rewards = np.array(
                            agent_data['reward'], dtype=np.float32)
                        values = np.array(
                            agent_data['value'], dtype=np.float32)
                        next_values = np.array(
                            agent_data['value'][1:] + [0], dtype=np.float32)

                        td_target = rewards + self.config.gamma * next_values
                        td_delta = td_target - values
                        advs = []
                        adv = 0
                        for delta in td_delta[::-1]:
                            adv = self.config.gamma * self.config.gae_lambda * adv + delta
                            advs.append(adv)
                        advs.reverse()
                        advantages = np.array(advs, dtype=np.float32)

                        self.replay_buffer.push({
                            'state': {'observation': obs, 'action_mask': mask},
                            'action': actions,
                            'adv': advantages,
                            'target': td_target
                        })
                    except:
                        continue

            # 更新阶段和迭代次数的标记，判断是否继续循环
            is_bc_stage = self.learner.get_BC_ongoing()
            iteration = self.learner.get_iteration()
