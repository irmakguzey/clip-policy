# calculate mean and std of dataset actions

import numpy as np
import torch

# TODO: May need to change when data is sequence based???
def get_act_mean_std(dataset):
    # print(dataset[0][1].shape)
    act_dim = dataset[0][1].shape[1]
    # print(act_dim)
    act_mean = torch.zeros(act_dim)
    act_std = torch.zeros(act_dim)
    count = 0
    for dataset in dataset.datasets:
        for label in dataset.labels:
            label = torch.from_numpy(label)
            # print(label.shape)
            act_mean += label
            act_std += label**2
            count += 1

    act_mean /= count
    act_std /= count
    act_std = torch.sqrt(act_std - act_mean**2)
    return act_mean, act_std
