"""Dataset utilities for RL-SFANC training and evaluation."""

import os

import numpy as np
import pandas as pd
import torchaudio
from torch.utils.data import Dataset


def minmaxscaler(data):
    data_min = data.min()
    data_max = data.max()
    return data / (data_max - data_min)


class MyNoiseDataset(Dataset):
    """Loads noise waveforms and (optional) evaluation labels from a CSV file."""

    def __init__(self, folder, annotations_file):
        self.folder = folder
        self.annotations_file = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, index):
        audio_sample_path = self.annotations_file.iloc[index, 1]
        label = np.array(self.annotations_file.iloc[index, 2]).astype(np.float32)
        signal, _ = torchaudio.load(os.path.join(self.folder, audio_sample_path))
        signal = minmaxscaler(signal)
        return signal, label
