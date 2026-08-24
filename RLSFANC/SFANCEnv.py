"""
Gymnasium-style MDP wrapper for RL-SFANC training.

State  : a 0.5 s noise frame (time-domain waveform).
Action : index of a pre-trained control filter, A = {1, ..., K}.
Reward : provided by `reward_fn(obs, action)`. In the paper this is the
         noise-reduction (NR) level obtained by applying the selected filter.

This release includes only the RL interaction loop. The acoustic / ANC
backend that computes residual error and NR is not included; pass your
own `reward_fn` following Section 3.3.1 of the paper.
"""

import os
import random

import numpy as np
import torchaudio
from gymnasium.spaces import Box, Discrete

from utils.dataset_utils import minmaxscaler


class SFANCEnv:
    """
    Frame-independent SFANC environment.

    Each step samples one noise frame, selects a filter (action), and receives
    a scalar reward. Episodes are terminated after every step; `episode_length`
    consecutive rewards are summed for logging.
    """

    def __init__(
        self,
        noise_dataset_path,
        reward_fn,
        fs=16000,
        frame_duration=0.5,
        noise_bound=1,
        episode_length=10,
        num_actions=7,
    ):
        if reward_fn is None:
            raise ValueError(
                "reward_fn is required. It should map (observation, action) to a "
                "scalar reward (NR in dB in the paper). The ANC backend is not released."
            )

        frame_len = int(frame_duration * fs)
        self.observation_space = Box(low=-noise_bound, high=noise_bound, shape=(1, frame_len), dtype=np.float32)
        self.action_space = Discrete(num_actions)

        self.fs = fs
        self.frame_len = frame_len
        self.reward_fn = reward_fn
        self.episode_length = episode_length

        file_names = os.listdir(noise_dataset_path)
        self.dataset = [os.path.join(noise_dataset_path, f) for f in file_names]

        self.current_obs = None
        self.test_rewards = np.zeros(episode_length, dtype=np.float32)
        self.current_step = 0

    def reset(self):
        noise_path = random.choice(self.dataset)
        waveform, _ = torchaudio.load(noise_path)
        waveform = waveform[0]
        if waveform.numel() < self.frame_len:
            raise ValueError(f"Noise file is shorter than one frame: {noise_path}")
        waveform = waveform[: self.frame_len]
        waveform = minmaxscaler(waveform)
        self.current_obs = waveform.unsqueeze(0).numpy().astype(np.float32)
        return self.current_obs, {}

    def step(self, action):
        reward = float(self.reward_fn(self.current_obs, int(action)))

        self.test_rewards[self.current_step] = reward
        self.current_step += 1

        # Frames are treated as independent, so each step ends an MDP transition.
        terminated = True
        next_obs = None

        if self.current_step % self.episode_length == 0:
            truncated = True
            info = {"episode_return": np.sum(self.test_rewards).item()}
            self.test_rewards = np.zeros(self.episode_length, dtype=np.float32)
            self.current_step = 0
        else:
            truncated = False
            info = {}

        return next_obs, reward, terminated, truncated, info
