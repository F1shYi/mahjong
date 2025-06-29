from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner
from config import Config
import os
from typing import List
import json
import random
import yaml


def get_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return Config(**config)


def get_curriculum_data(config: Config):
    if config.use_curriculum == False:
        return None
    json_files = os.listdir(config.curriculum_data_folder)
    full_shanten_tilewalls: List[List[str]] = [[]
                                               for _ in range(len(json_files))]
    for json_file in json_files:
        shanten_num = int(json_file.split('.')[0][-1])
        with open(os.path.join(config.curriculum_data_folder, json_file), 'r') as f:
            tilewalls = json.load(f)
            full_tilewalls = tilewalls[str(shanten_num)]
            full_shanten_tilewalls[shanten_num].extend(
                tilewall['tilewall'] for tilewall in full_tilewalls)
    for shanten_tilewalls in full_shanten_tilewalls:
        random.shuffle(shanten_tilewalls)

    return full_shanten_tilewalls


if __name__ == '__main__':

    config = get_config("configs/PPO_only_cnn.yaml")
    full_shanten_tilewalls = get_curriculum_data(config)

    replay_buffer = ReplayBuffer(
        config.replay_buffer_size, config.replay_buffer_episode)

    learner = Learner(config, replay_buffer)
    actors = []
    for i in range(config.num_actors):
        actor = Actor(config, replay_buffer,
                      i, learner, full_shanten_tilewalls)
        actors.append(actor)

    for actor in actors:
        actor.start()
    learner.start()

    for actor in actors:
        actor.join()
    learner.terminate()
