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
        self.learner = learner  # Need to know the global training iteration
        print(f"Actor {self.actor_id} initialized")
        # slicing
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
                    f"Actor {self.actor_id} using {shanten_num} shanten data, length = {len(shanten_tilewalls)}")

    def get_tilewalls(self) -> str:
        if self.config.use_curriculum == False:
            return ''
        current_iteration = self.learner.get_iteration()
        if current_iteration >= self.config.curriculum_iterations[self.current_shanten]:
            self.current_shanten += 1
            self.shanten_tilewalls_idx = 0
            print(
                f"Actor {self.actor_id} switch to {self.current_shanten} shanten data")
        tilewall = self.shanten_tilewalls[self.current_shanten][self.shanten_tilewalls_idx]
        self.shanten_tilewalls_idx += 1
        self.shanten_tilewalls_idx = self.shanten_tilewalls_idx % len(
            self.shanten_tilewalls[self.current_shanten])
        return tilewall

    def run(self):
        torch.set_num_threads(1)

        # connect to model pool
        model_pool = ModelPoolClient(self.config.model_pool_name)

        # create network model
        model = CNNModel()

        # load initial model
        version = model_pool.get_latest_model()
        state_dict = model_pool.load_model(version)
        model.load_state_dict(state_dict)

        # collect data
        env = MahjongGBEnv(config={'agent_clz': FeatureAgent})
        # all four players use the latest model
        policies = {player: model for player in env.agent_names}

        while self.learner.get_iteration() < self.config.max_iterations:
            # update model
            latest = model_pool.get_latest_model()
            if latest['id'] > version['id']:
                state_dict = model_pool.load_model(latest)
                model.load_state_dict(state_dict)
                version = latest

            # run one episode and collect data
            obs = env.reset(tileWall=self.get_tilewalls())
            episode_data = {agent_name: {
                'state': {
                    'observation': [],
                    'action_mask': []
                },
                'action': [],
                'reward': [],
                'value': []
            } for agent_name in env.agent_names}
            done = False
            while not done:
                # each player take action
                actions = {}
                values = {}
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
                    model.train(False)  # Batch Norm inference mode
                    with torch.no_grad():
                        logits, value = model(state)
                        action_dist = torch.distributions.Categorical(
                            logits=logits)
                        action = action_dist.sample().item()
                        value = value.item()
                    actions[agent_name] = action
                    values[agent_name] = value
                    agent_data['action'].append(actions[agent_name])
                    agent_data['value'].append(values[agent_name])
                # interact with env
                next_obs, rewards, done, status = env.step(actions)
                for agent_name in rewards:
                    episode_data[agent_name]['reward'].append(
                        rewards[agent_name])
                obs = next_obs
            # print(self.name, 'Episode', episode, 'Model',
            #       latest['id'], 'Reward', rewards)

            # postprocessing episode data for each agent
            for agent_name, agent_data in episode_data.items():
                try:
                    if len(agent_data['action']) < len(agent_data['reward']):
                        agent_data['reward'].pop(0)
                    obs = np.stack(agent_data['state']['observation'])
                    mask = np.stack(agent_data['state']['action_mask'])
                    actions = np.array(agent_data['action'], dtype=np.int64)
                    rewards = np.array(agent_data['reward'], dtype=np.float32)
                    values = np.array(agent_data['value'], dtype=np.float32)
                    next_values = np.array(
                        agent_data['value'][1:] + [0], dtype=np.float32)

                    td_target = rewards + next_values * self.config.gamma
                    td_delta = td_target - values
                    advs = []
                    adv = 0
                    for delta in td_delta[::-1]:
                        adv = self.config.gamma * \
                            self.config.gae_lambda * adv + delta
                        advs.append(adv)  # GAE
                    advs.reverse()
                    advantages = np.array(advs, dtype=np.float32)

                    # send samples to replay_buffer (per agent)
                    self.replay_buffer.push({
                        'state': {
                            'observation': obs,
                            'action_mask': mask
                        },
                        'action': actions,
                        'adv': advantages,
                        'target': td_target
                    })
                except:
                    continue
