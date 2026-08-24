"""Evaluate a trained Q-network on a labelled test set (labels are for evaluation only)."""

import argparse

from RLSFANC.SFANCEnv import SFANCEnv
from RLSFANC.Algos import DQN
from RLSFANC.Networks import CNNRes


def dummy_reward(observation, action):
    """Placeholder reward; testing only uses the Q-network and labelled data."""
    return 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained DQN for RL-SFANC.")
    parser.add_argument("--exp-name", type=str, default="dqn")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--num-actions", type=int, default=7)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--test-labels", type=str, required=True)
    parser.add_argument("--load-model", type=str, required=True)
    parser.add_argument("--save-folder", type=str, default="./models/dqn/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    env = SFANCEnv(
        noise_dataset_path=args.test_data,
        reward_fn=dummy_reward,
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
        load_model=args.load_model,
        save_folder=args.save_folder,
    )

    acc = agent.test()
    print("Test accuracy: {:.4f}".format(acc))
