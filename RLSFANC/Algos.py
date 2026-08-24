"""
Deep Q-Network (DQN) for training the CNN in RL-SFANC.

The Q-network takes a noise frame as input and outputs Q-values over
pre-trained control filters. Because only the immediate noise-reduction
reward is used, the discount factor is set to 0 and the TD target equals
the instantaneous reward (see the paper, Section 3.3.2).

References:
- Mnih et al., Human-level control through deep reinforcement learning, Nature, 2015.
- CleanRL: https://docs.cleanrl.dev/rl-algorithms/dqn/
"""

import os
import random
import datetime
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from stable_baselines3.common.buffers import ReplayBuffer

from utils.dataset_utils import MyNoiseDataset


class DQN:
    """DQN trainer for the SFANC filter-selection policy."""

    def __init__(
        self,
        env,
        q_network_class,
        exp_name="dqn",
        seed=1,
        cuda=0,
        test_dataset=None,
        test_labels_file=None,
        test_frequency=1000,
        learning_rate=2.5e-4,
        buffer_size=10000,
        rb_optimize_memory=False,
        batch_size=128,
        start_e=1,
        end_e=0.05,
        exploration_fraction=0.5,
        train_frequency=10,
        load_model=None,
        write_frequency=100,
        save_frequency=10000,
        save_folder="./models/dqn/",
    ):
        self.exp_name = exp_name
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
        self.env = env

        self.q_network = q_network_class(num_classes=env.action_space.n).to(self.device)
        if load_model is not None:
            self.q_network.load_state_dict(torch.load(load_model, map_location=self.device))

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(
            buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            optimize_memory_usage=rb_optimize_memory,
            handle_timeout_termination=False,
        )

        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction
        self.batch_size = batch_size
        self.train_frequency = train_frequency

        run_name = "{}-{}-{}".format(
            exp_name, seed, datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S")
        )
        os.makedirs("./runs/", exist_ok=True)
        self.writer = SummaryWriter(os.path.join("./runs/", run_name))
        self.write_frequency = write_frequency

        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.best_episodic_return = float("-inf")
        self.save_frequency = save_frequency

        if test_dataset is None:
            self.test_dataset_loader = None
        else:
            self.test_frequency = test_frequency
            test_dataset = MyNoiseDataset(test_dataset, test_labels_file)
            self.test_dataset_loader = DataLoader(test_dataset, batch_size=1)
            self.best_accuracy = float("-inf")

    def linear_schedule(self, duration, t):
        """Linear interpolation from start_e to end_e over `duration` steps."""
        slope = (self.end_e - self.start_e) / duration
        return max(slope * t + self.start_e, self.end_e)

    def learn(self, total_timesteps=500000, learning_starts=10000):
        obs, _ = self.env.reset()
        for global_step in range(1, total_timesteps + 1):
            epsilon = self.linear_schedule(self.exploration_fraction * total_timesteps, global_step)

            if random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                q_value = self.q_network(torch.Tensor(obs).unsqueeze(0).to(self.device))
                action = torch.argmax(q_value).cpu().numpy()

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            if truncated:
                episodic_return = info["episode_return"]
                if episodic_return >= self.best_episodic_return:
                    self.save(indicator="best-return")
                    self.best_episodic_return = episodic_return

                print(f"global_step={global_step}, episodic_return={episodic_return}")
                self.writer.add_scalar("charts/episodic_return", episodic_return, global_step)

                if self.test_dataset_loader is not None and global_step % self.test_frequency == 0:
                    acc = self.test()
                    if acc >= self.best_accuracy:
                        self.save(indicator="best-accuracy")
                        self.best_accuracy = acc
                    print(f"global_step={global_step}, accuracy={acc}")
                    self.writer.add_scalar("charts/test_accuracy", acc, global_step)

                if global_step % self.save_frequency == 0:
                    self.save(indicator="{}k".format(int(global_step / 1000)))

            self.replay_buffer.add(obs, next_obs, action, reward, terminated, info)

            if terminated:
                obs, _ = self.env.reset()
            else:
                obs = next_obs

            if global_step > learning_starts and global_step % self.train_frequency == 0:
                self.optimize(global_step)

        self.save(indicator="final")
        self.writer.close()

    def optimize(self, global_step):
        data = self.replay_buffer.sample(self.batch_size)
        # With gamma = 0, the TD target is the instantaneous reward.
        td_target = data.rewards.flatten()
        old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)

        if global_step % self.write_frequency == 0:
            self.writer.add_scalar("losses/td_loss", loss, global_step)
            self.writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, indicator="best"):
        torch.save(
            self.q_network.state_dict(),
            os.path.join(self.save_folder, "q_network-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)),
        )

    def test(self):
        """Evaluate filter-selection accuracy on a labelled test set (labels are used only for evaluation)."""
        self.q_network.eval()
        eval_acc = 0
        for input, label in self.test_dataset_loader:
            input, label = input.to(self.device), label.to(self.device)
            prediction = self.q_network(input)
            pred = torch.argmax(prediction).item()
            eval_acc += 1 if pred == label.item() else 0
        self.q_network.train()
        return eval_acc / len(self.test_dataset_loader)
