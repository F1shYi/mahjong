from multiprocessing import Process
import time
import numpy as np
import torch
from torch.nn import functional as F
import os
from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import CNNModel
from env import ERROR, HASWINNER, NOWINNER, MahjongGBEnv
from feature import FeatureAgent
from torch.utils.tensorboard.writer import SummaryWriter


def eval_one_episode(model: CNNModel, env: MahjongGBEnv):

    obs = env.reset()
    done = False
    episode_length = 0
    while not done:
        episode_length += 1
        actions = {}
        values = {}
        for agent_name in obs:
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
    return episode_length, status


def eval(model, env: MahjongGBEnv, episodes=100):

    episode_lengths = []
    statuses = []
    for _ in range(episodes):
        episode_length, status = eval_one_episode(model, env)
        episode_lengths.append(episode_length)
        statuses.append(status)

    has_winner_count = 0
    no_winner_count = 0
    error_count = 0
    valid_episode_length = []

    for episode_length, status in zip(episode_lengths, statuses):
        if status == HASWINNER:
            has_winner_count += 1
            valid_episode_length.append(episode_length)
        elif status == NOWINNER:
            no_winner_count += 1
            valid_episode_length.append(episode_length)
        elif status == ERROR:
            error_count += 1

    avg_episode_length = np.mean(valid_episode_length)
    has_winner_percentage = has_winner_count / episodes
    no_winner_percentage = no_winner_count / episodes
    error_percentage = error_count / episodes

    return avg_episode_length, has_winner_percentage, no_winner_percentage, error_percentage


class Learner(Process):

    def __init__(self, config, replay_buffer):

        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.output_dir = self.config['output_path']
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.output_dir = os.path.join(self.output_dir, self.timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        self.model_dir = os.path.join(self.output_dir, 'model')
        os.makedirs(self.model_dir, exist_ok=True)
        self.log_dir = os.path.join(self.output_dir, 'log')
        os.makedirs(self.log_dir, exist_ok=True)
        self.reset_metrics()

    def reset_metrics(self):
        self.total_loss = 0
        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        self.mean_ratio = 0
        self.count = 0

    def log_metrics(self, iterations):
        if self.count > 0:
            self.writer.add_scalar(
                'Loss/total', self.total_loss / self.count, iterations)
            self.writer.add_scalar(
                'Loss/policy', self.policy_loss / self.count, iterations)
            self.writer.add_scalar(
                'Loss/value', self.value_loss / self.count, iterations)
            self.writer.add_scalar(
                'Loss/entropy', self.entropy_loss / self.count, iterations)
            self.writer.add_scalar(
                'PPO/mean_ratio', self.mean_ratio / self.count, iterations)
        self.reset_metrics()

    def run(self):
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # create model pool
        model_pool = ModelPoolServer(
            self.config['model_pool_size'], self.config['model_pool_name'])

        # initialize model params
        device = torch.device(self.config['device'])
        model = CNNModel()

        # send to model pool
        # push cpu-only tensor to model_pool
        model_pool.push(model.state_dict())
        model = model.to(device)

        # training
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])

        # wait for initial samples
        while self.replay_buffer.size() < self.config['min_sample']:
            time.sleep(0.1)

        iterations = 0
        while True:
            # sample batch
            batch = self.replay_buffer.sample(self.config['batch_size'])
            obs = torch.tensor(batch['state']['observation']).to(device)
            mask = torch.tensor(batch['state']['action_mask']).to(device)
            states = {
                'observation': obs,
                'action_mask': mask
            }
            actions = torch.tensor(batch['action']).unsqueeze(-1).to(device)
            advs = torch.tensor(batch['adv']).to(device)
            targets = torch.tensor(batch['target']).to(device)

            # print('Iteration %d, replay buffer in %d out %d' % (
            #     iterations, self.replay_buffer.stats['sample_in'], self.replay_buffer.stats['sample_out']))

            # calculate PPO loss
            model.train(True)  # Batch Norm training mode
            old_logits, _ = model(states)
            old_probs = F.softmax(old_logits, dim=1).gather(1, actions)
            old_log_probs = torch.log(old_probs + 1e-8).detach()
            for _ in range(self.config['epochs']):
                logits, values = model(states)
                action_dist = torch.distributions.Categorical(logits=logits)
                probs = F.softmax(logits, dim=1).gather(1, actions)
                log_probs = torch.log(probs + 1e-8)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advs
                surr2 = torch.clamp(
                    ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = torch.mean(
                    F.mse_loss(values.squeeze(-1), targets))
                entropy_loss = -torch.mean(action_dist.entropy())
                loss = policy_loss + \
                    self.config['value_coeff'] * value_loss + \
                    self.config['entropy_coeff'] * entropy_loss

                self.total_loss += loss.item()
                self.policy_loss += policy_loss.item()
                self.value_loss += value_loss.item()
                self.entropy_loss += entropy_loss.item()
                self.mean_ratio += ratio.mean().item()
                self.count += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if iterations % self.config['log_interval'] == 0:
                self.log_metrics(iterations)

            # push new model
            model = model.to('cpu')
            # push cpu-only tensor to model_pool
            model_pool.push(model.state_dict())
            model = model.to(device)

            # eval and save checkpoints

            if iterations % self.config['ckpt_save_interval'] == 0:
                print("saving and evaluating")
                path = os.path.join(self.model_dir, 'model_%d.pt' % iterations)
                torch.save(model.state_dict(), path)
            if iterations % self.config['eval_interval'] == 0:
                model = model.to('cpu')
                avg_episode_length, has_winner_percentage, no_winner_percentage, error_percentage = eval(
                    model, MahjongGBEnv(config={'agent_clz': FeatureAgent}))

                self.writer.add_scalar(
                    'Eval/avg_episode_length', avg_episode_length, iterations)
                self.writer.add_scalar(
                    'Eval/win_rate', has_winner_percentage, iterations)
                self.writer.add_scalar(
                    'Eval/no_winner_rate', no_winner_percentage, iterations)
                self.writer.add_scalar(
                    'Eval/error_rate', error_percentage, iterations)

                model = model.to(device)
            iterations += 1
        self.writer.close()
