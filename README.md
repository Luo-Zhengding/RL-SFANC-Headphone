# RL-SFANC: Reinforcement Learning-based Selective Fixed-Filter Active Noise Control

This repository provides the open-source **reinforcement learning algorithm and training code** for our paper:

> Luo, Z., Ma, H., Wang, B., Shi, D., & Gan, W. S. (2026). Reinforcement learning-based selective fixed-filter active noise control (RL-SFANC): From theory to real-time headphone implementation. *Signal Processing*, 110695.  
> [https://doi.org/10.1016/j.sigpro.2026.110695](https://doi.org/10.1016/j.sigpro.2026.110695)

This release contains the DQN-based CNN training pipeline of RL-SFANC (Q-network, DQN trainer, and Gymnasium-style RL loop). The acoustic / noise-cancellation backend (control-filter application, residual-error computation, and the real-time headphone controller) is **not** included.

Researchers interested in this work are welcome to use the code and cite the paper.

## Highlights

- **Label-free CNN training.** A DQN trains the CNN through interaction with the acoustic environment, using noise reduction as the reward. This removes the need for labelled noise data and allows the agent to explore different filter selections.
- **Handling discrete, non-differentiable actions.** The CNN outputs a discrete filter index, which cannot be trained by back-propagating the residual error. RL optimizes the selection policy in a discrete action space and bypasses this issue.
- **Real-time headphone implementation.** RL-SFANC is deployed on an ANC headphone with a dual-rate architecture, enabling delayless noise cancellation.
- **Transferability across acoustic paths.** A CNN trained on synthetic paths can be used on real headphone paths without retraining; only the pre-trained control filters need to be updated.

## Repository Structure

```
RL-SFANC/
├── RLSFANC/
│   ├── Algos.py          # DQN training algorithm
│   ├── Networks.py       # Lightweight 1D CNN (Q-network) for filter selection
│   └── SFANCEnv.py       # Gymnasium-style RL environment (MDP loop)
├── utils/
│   └── dataset_utils.py  # Noise-frame dataset loader
├── run.py                # Train the DQN agent
├── test.py               # Evaluate a trained Q-network
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

Implement `noise_reduction_reward(observation, action)` in `run.py` so that it returns the noise-reduction level (dB) of the selected filter, as defined in the paper. Then run:

```bash
python run.py --train-data /path/to/training_noise_frames --test-data /path/to/testing_noise_frames --test-labels /path/to/testing_label.csv
```

Default hyperparameters follow the paper (160,000 training steps, learning rate `2.5e-4`, replay buffer 160,000, batch size 128, epsilon from 1.0 to 0.05).

## Evaluation

```bash
python test.py --test-data /path/to/testing_noise_frames --test-labels /path/to/testing_label.csv --load-model /path/to/q_network.pth
```

Labels are used only for evaluation, not for training.

## Citation

If you find this work useful, please cite:

```bibtex
@article{luo2026reinforcement,
  title={Reinforcement learning-based selective fixed-filter active noise control (RL-SFANC): From theory to real-time headphone implementation},
  author={Luo, Zhengding and Ma, Haozhe and Wang, Boxiang and Shi, Dongyuan and Gan, Woon-Seng},
  journal={Signal Processing},
  pages={110695},
  year={2026},
  publisher={Elsevier}
}
```

Thank you for your interest!
