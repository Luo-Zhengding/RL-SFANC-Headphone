"""
Train the DQN agent for RL-SFANC.

The acoustic / ANC backend is not included in this release. Implement
`noise_reduction_reward(observation, action)` using your own simulator
so that the returned scalar is the noise-reduction level (dB), as in
Eq. (12) of the paper.
"""

import argparse

from RLSFANC.SFANCEnv import SFANCEnv
from RLSFANC.Algos import DQN
from RLSFANC.Networks import CNNRes


def noise_reduction_reward(observation, action):
    """
    Reward function of RL-SFANC.

    Parameters
    ----------
    observation : np.ndarray, shape (1, L)
        Current noise frame (state).
    action : int
        Index of the selected pre-trained control filter.

    Returns
    -------
    float
        Noise reduction (NR) in dB after applying the selected filter.
    """
    raise NotImplementedError(
        "Please implement the NR reward with your own ANC backend. "
        "The noise-cancellation code is not included in this release. "
        "See Section 3.3.1 of the paper for the reward definition."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN for RL-SFANC.")
    parser.add_argument("--exp-name", type=str, default="dqn")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--buffer-size", type=int, default=160000)
    parser.add_argument("--rb-optimize-memory", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-e", type=float, default=1.0)
    parser.add_argument("--end-e", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.5)
    parser.add_argument("--train-frequency", type=int, default=10)
    parser.add_argument("--write-frequency", type=int, default=100)
    parser.add_argument("--save-folder", type=str, default="./models/dqn/")
    parser.add_argument("--total-timesteps", type=int, default=160000)
    parser.add_argument("--learning-starts", type=int, default=10000)
    parser.add_argument("--episode-length", type=int, default=10)
    parser.add_argument("--num-actions", type=int, default=7)
    parser.add_argument("--test-frequency", type=int, default=1000)
    parser.add_argument("--save-frequency", type=int, default=1000)
    parser.add_argument("--train-data", type=str, required=True, help="Folder of training noise frames.")
    parser.add_argument("--test-data", type=str, default=None, help="Folder of test noise frames.")
    parser.add_argument("--test-labels", type=str, default=None, help="CSV file of test labels (evaluation only).")
    parser.add_argument("--load-model", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    env = SFANCEnv(
        noise_dataset_path=args.train_data,
        reward_fn=noise_reduction_reward,
        episode_length=args.episode_length,
        num_actions=args.num_actions,
    )

    agent = DQN(
        env,
        q_network_class=CNNRes,
        exp_name=args.exp_name,
        seed=args.seed,
        cuda=args.cuda,
        test_dataset=args.test_data,
        test_labels_file=args.test_labels,
        test_frequency=args.test_frequency,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        rb_optimize_memory=args.rb_optimize_memory,
        batch_size=args.batch_size,
        start_e=args.start_e,
        end_e=args.end_e,
        exploration_fraction=args.exploration_fraction,
        train_frequency=args.train_frequency,
        load_model=args.load_model,
        write_frequency=args.write_frequency,
        save_frequency=args.save_frequency,
        save_folder=args.save_folder,
    )

    agent.learn(total_timesteps=args.total_timesteps, learning_starts=args.learning_starts)
    agent.save(indicator="final")
    print("DQN for RL-SFANC has been trained successfully.")
