import h5py
import numpy as np
import torch

from RL.Utils import Dataset

SL_theorem_Data = Dataset(mode="SL_theorem", save_path='./Database/SL/pretrain/theorem/E01.h5')
SL_param_Data = Dataset(mode="SL_param", save_path='./Database/SL/pretrain/param/E01.h5')
# dataset.createDataset(states=states, params=params)
# dataset.addDataset(state=states, param=params)
theorem_dataloader = SL_theorem_Data.loadDataset(1)
# param_dataloader = SL_param_Data.loadDataset(8)

all_sum = 0

for a, b in theorem_dataloader:
    all_sum += 1

print(all_sum)
