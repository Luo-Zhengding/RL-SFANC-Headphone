# RL-SFANC: Reinforcement Learning-based Selective Fixed-Filter Active Noise Control

This repository provides the open-source **reinforcement learning algorithm and training code** for our paper:

> Luo, Z., Ma, H., Wang, B., Shi, D., & Gan, W. S. (2026). Reinforcement learning-based selective fixed-filter active noise control (RL-SFANC): From theory to real-time headphone implementation. *Signal Processing*, 110695.  
> [https://doi.org/10.1016/j.sigpro.2026.110695](https://doi.org/10.1016/j.sigpro.2026.110695)

The released code covers the DQN-based CNN training pipeline of RL-SFANC (environment, networks, and training scripts). It does **not** include the real-time headphone controller implementation.

Researchers interested in this work are welcome to use the code and cite the paper.

## Highlights

- **Label-free CNN training.** A DQN trains the CNN through interaction with the acoustic environment, using noise reduction as the reward. This removes the need for labelled noise data and allows the agent to explore different filter selections.
- **Handling discrete, non-differentiable actions.** The CNN outputs a discrete filter index, which cannot be trained by back-propagating the residual error. RL optimizes the selection policy in a discrete action space and bypasses this issue.
- **Real-time headphone implementation.** RL-SFANC is deployed on an ANC headphone with a dual-rate architecture, enabling delayless noise cancellation.
- **Transferability across acoustic paths.** A CNN trained on synthetic paths can be used on real headphone paths without retraining; only the pre-trained control filters need to be updated.

## Repository Structure

```
2.DQN_SFANC/
├── RLSFANC/
│   ├── Algos.py          # DQN / Categorical DQN training algorithms
│   ├── Networks.py       # Lightweight 1D CNN (Q-network) for filter selection
│   └── SFANCEnv.py       # Gymnasium-style SFANC environment
├── utils/                # Dataset, noise, filter, and ANC utilities
├── run.py                # Train the DQN agent
├── run-cdqn.py           # Train the Categorical DQN agent
├── test.py               # Evaluate a trained CNN on the test set
└── run-many.sh           # Example multi-seed training commands
```

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
