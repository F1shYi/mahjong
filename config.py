from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    # replay buffer
    num_actors: int
    replay_buffer_size: int
    replay_buffer_episode: int

    # model pool
    model_pool_size: int
    model_pool_name: str

    # PPO Training
    gamma: float
    gae_lambda: float
    min_sample: int
    batch_size: int
    epochs: int
    clip: float
    lr: float
    value_coeff: float
    entropy_coeff: float
    device: str
    max_iterations: int

    # measured by iterations
    log_interval: int
    ckpt_save_interval: int
    eval_interval: int
    output_path: str

    # Curriculum Learning
    use_curriculum: bool
    curriculum_data_folder: str
    # if `use_curriculum`,
    # the model will be trained in the environment
    # which is reset by i-shanten initial conditions for `curriculum_iterations[i]` iterations
    # else ignored.
    curriculum_iterations: List[int]
