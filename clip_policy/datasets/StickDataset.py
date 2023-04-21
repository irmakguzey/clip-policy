import itertools
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset, Subset, ConcatDataset
from pathlib import Path
import numpy as np
import einops
from typing import Union, Callable, Optional, Sequence, List, Any
from tqdm import tqdm
import abc
from torch import default_generator, randperm
from torch._utils import _accumulate
from scipy.spatial.transform import Rotation as R
from PIL import Image
import torchvision
import json
import random

# import datetime
import datetime
import pickle

# create abstract Dataset class called StickDataset
import sys
import cv2


import liblzfse

from utils.r3D_semantic_dataset import load_depth
from utils.metrics import get_act_mean_std
from utils.traverse_data import iter_dir_for_traj_pths


from scipy.optimize import linear_sum_assignment


class BaseStickDataset(Dataset, abc.ABC):
    def __init__(self, traj_path, time_skip, time_offset, time_trim):
        super().__init__()
        self.traj_path = Path(traj_path)
        self.time_skip = time_skip
        self.time_offset = time_offset
        self.time_trim = time_trim
        self.img_pth = self.traj_path / "images"
        self.depth_pth = self.traj_path / "depths"
        self.conf_pth = self.traj_path / "confs"
        self.labels_pth = self.traj_path / "labels.json"
        # TODO: change the followibg line to the correct path
        self.detection_pth = self.traj_path / "detections.pkl"

        self.labels = json.load(self.labels_pth.open("r"))
        self.img_keys = sorted(self.labels.keys())

        # lable structure: {image_name: {'xyz' : [x,y,z], 'rpy' : [r, p, y], 'gripper': gripper}, ...}

        self.labels = np.array(
            [self.flatten_label(self.labels[k]) for k in self.img_keys]
        )

        # filter using time_skip and time_offset and time_trim. start from time_offset, skip time_skip, and remove last time_trim
        self.labels = self.labels[: -self.time_trim][self.time_offset :: self.time_skip]

        # filter keys using time_skip and time_offset and time_trim. start from time_offset, skip time_skip, and remove last time_trim
        self.img_keys = self.img_keys[: -self.time_trim][
            self.time_offset :: self.time_skip
        ]

    def flatten_label(self, label):
        # flatten label
        xyz = label["xyz"]
        rpy = label["rpy"]
        gripper = label["gripper"]
        return np.concatenate((xyz, rpy, np.array([gripper])))

    def __len__(self):
        return len(self.img_keys)

    def __getitem__(self, idx):
        # not implemented

        raise NotImplementedError


class StickDataset(BaseStickDataset, abc.ABC):
    def __init__(
        self, traj_path, traj_len, time_skip, time_offset, time_trim, traj_skip
    ):
        super().__init__(traj_path, time_skip, time_offset, time_trim)
        self.traj_len = traj_len
        self.traj_skip = traj_skip
        self.reformat_labels(self.labels)
        self.act_metrics = None

    def set_act_metrics(self, act_metrics):
        self.act_metrics = act_metrics

    def get_cost_from_clip(self, clip_embeddings):
        # clip_embeddings: (L, 512)
        # self.clip_embeddings: (M, N, 512)
        # M: length of the current trajectory, N: Number of objects detected in the given frame
        # L: number of words in the given query

        total_cost = 0
        for frame_embeddings in self.clip_embeddings:
            # calculate N x L cost matrix
            cost_matrix = torch.cdist(frame_embeddings, clip_embeddings, p=2)
            # use hungarian algorithm to find the best match
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # add the cost of the best match
            total_cost += cost_matrix[row_ind, col_ind].sum()
        return total_cost / len(self.clip_embeddings)

    def reformat_labels(self, labels):
        # reformat labels to be delta xyz, delta rpy, next gripper state
        new_labels = np.zeros_like(labels)
        new_img_keys = []

        for i in range(len(labels) - 1):
            if i == 0:
                current_label = labels[i]
                next_label = labels[i + 1]
            else:
                next_label = labels[i + 1]

            current_matrix = np.eye(4)
            r = R.from_euler("xyz", current_label[3:6], degrees=False)
            current_matrix[:3, :3] = r.as_matrix()
            current_matrix[:3, 3] = current_label[:3]

            next_matrix = np.eye(4)
            r = R.from_euler("xyz", next_label[3:6], degrees=False)
            next_matrix[:3, :3] = r.as_matrix()
            next_matrix[:3, 3] = next_label[:3]

            delta_matrix = np.linalg.inv(current_matrix) @ next_matrix
            delta_xyz = delta_matrix[:3, 3]
            delta_r = R.from_matrix(delta_matrix[:3, :3])
            delta_rpy = delta_r.as_euler("xyz", degrees=False)

            del_gripper = next_label[6] - current_label[6]
            xyz_norm = np.linalg.norm(delta_xyz)
            rpy_norm = np.linalg.norm(delta_r.as_rotvec())

            if xyz_norm < 0.01 and rpy_norm < 0.008 and abs(del_gripper) < 0.05:
                # drop this label and corresponding image_key since the delta is too small (basically the same image)
                continue

            new_labels[i] = np.concatenate(
                (delta_xyz, delta_rpy, np.array([next_label[6]]))
            )
            new_img_keys.append(self.img_keys[i])
            current_label = next_label

        # remove labels with all 0s
        new_labels = new_labels[new_labels.sum(axis=1) != 0]
        assert len(new_labels) == len(new_img_keys)
        self.labels = new_labels
        self.img_keys = new_img_keys

    def load_labels(self, idx):
        # load labels with window size of traj_len, starting from idx and moving window by traj_skip
        labels = self.labels[
            idx * self.traj_skip : idx * self.traj_skip + self.traj_len
        ]
        # normalize labels
        if self.act_metrics is not None:
            labels = (labels - self.act_metrics["mean"].numpy()) / self.act_metrics[
                "std"
            ].numpy()
        return labels

    def load_detection_labels(self):
        # TODO: load detection labels with the help from Irmak
        raise NotImplementedError

    def get_img_pths(self, idx):
        # get image paths with window size of traj_len, starting from idx and moving window by traj_skip
        img_keys = self.img_keys[
            idx * self.traj_skip : idx * self.traj_skip + self.traj_len
        ]
        img_pths = [self.img_pth / k for k in img_keys]
        return img_pths

    def __len__(self):
        return (len(self.img_keys) - self.traj_len) // self.traj_skip + 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError()
        return None, self.load_labels(idx)


class ImageStickDataset(StickDataset):
    def __init__(
        self,
        traj_path,
        traj_len,
        time_skip,
        time_offset,
        time_trim,
        traj_skip,
        img_size,
        pre_load=False,
        transforms=None,
    ):
        super().__init__(
            traj_path, traj_len, time_skip, time_offset, time_trim, traj_skip
        )
        self.img_size = img_size
        self.pre_load = pre_load
        self.transforms = transforms
        self.preprocess_img_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.img_size),
                torchvision.transforms.ToTensor(),
            ]
        )
        if self.pre_load:
            self.imgs = self.load_imgs()

    def load_imgs(self):
        # load images in uint8 with window size of traj_len, starting from idx and moving window by traj_skip
        imgs = []

        for key in tqdm(self.img_keys):
            img = Image.open(str(self.img_pth / key))
            img = self.preprocess_img_transforms(img)
            imgs.append(img)
        # add a nex axis at the beginning
        imgs = torch.stack(imgs, dim=0)
        return imgs

    def __getitem__(self, idx):
        _, labels = super().__getitem__(idx)

        if self.pre_load:
            imgs = self.imgs[
                idx * self.traj_skip : idx * self.traj_skip + self.traj_len
            ]
        else:
            imgs = []
            for key in self.img_keys[
                idx * self.traj_skip : idx * self.traj_skip + self.traj_len
            ]:
                img = Image.open(str(self.img_pth / key))
                img = self.preprocess_img_transforms(img)
                imgs.append(img)
            # add a nex axis at the beginning
            imgs = torch.stack(imgs, dim=0)

        if self.transforms:
            imgs = self.transforms(imgs)

        return imgs, labels


class DiscreteImageStickDataset(ImageStickDataset):
    def discretize_labels(self, discretizer):
        # discretize labels into 5 bins
        self.discretizer = discretizer
        self.true_labels = self.labels
        self.labels = self.discretizer.encode_into_latent(torch.Tensor(self.labels))


class DepthStickDataset(ImageStickDataset):
    def __init__(
        self,
        traj_path,
        traj_len,
        time_skip,
        time_offset,
        time_trim,
        traj_skip,
        img_size,
        pre_load=False,
        transforms=None,
    ):
        if self.pre_load:
            self.drop_nan()
        super().__init__(
            traj_path,
            traj_len,
            time_skip,
            time_offset,
            time_trim,
            traj_skip,
            img_size,
            pre_load,
            transforms,
        )

    def load_imgs(self):
        # load images in uint8 with window size of traj_len, starting from idx and moving window by traj_skip
        imgs = []
        global_max = 0
        for key in tqdm(self.img_keys):
            img = load_depth(str(self.depth_pth / (key[:-4] + ".depth")))
            img = cv2.resize(img, (self.img_size, self.img_size)) / 4
            # add a new axis at the 0th dimension
            img = img[None, :, :]
            img = torch.from_numpy(img)
            imgs.append(img)
        # add a nex axis at the beginning
        imgs = torch.stack(imgs, dim=0)
        return imgs

    def drop_nan(self):
        # reformat labels to be delta xyz, delta rpy, next gripper state
        new_labels = np.zeros_like(self.labels)
        new_imgs = []
        new_img_keys = []
        idxs_dropped = []
        for i, img in enumerate(self.imgs):
            # count number of nan values in image
            no_of_nans = torch.isnan(img).sum()
            # if there are more than 5% nan values, drop the image
            if no_of_nans > 0.05 * img.shape[1] * img.shape[2]:
                idxs_dropped.append(i)
                continue
            else:
                # fill nan values with 5, a placeholder value
                img[torch.isnan(img)] = 0
                # print('no of nans: ', torch.isnan(img).sum())
                new_imgs.append(img)
                new_img_keys.append(self.img_keys[i])
                new_labels[i] = self.labels[i]

        # remove labels with all 0s
        if len(idxs_dropped) > 0:
            print(f"dropped {idxs_dropped} images")
        new_labels = new_labels[new_labels.sum(axis=1) != 0]
        assert len(new_labels) == len(new_img_keys)
        assert len(new_labels) == len(new_imgs)
        self.imgs = torch.stack(new_imgs, dim=0)
        self.labels = new_labels
        self.img_keys = new_img_keys


class ImageDepthStickDataset(DepthStickDataset, DiscreteImageStickDataset):
    def __init__(
        self,
        traj_path,
        traj_len,
        time_skip,
        time_offset,
        time_trim,
        traj_skip,
        img_size,
        pre_load=False,
        img_transforms=None,
        depth_transforms=None,
        concat_img_depth=False,
    ):
        StickDataset.__init__(
            self, traj_path, traj_len, time_skip, time_offset, time_trim, traj_skip
        )
        self.img_size = img_size
        self.pre_load = pre_load
        self.img_transforms = img_transforms
        self.depth_transforms = depth_transforms
        self.concat_img_depth = concat_img_depth
        self.preprocess_img_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((self.img_size, self.img_size)),
                torchvision.transforms.ToTensor(),
            ]
        )

        if self.pre_load:
            self.imgs = DepthStickDataset.load_imgs(self)
            self.drop_nan()
            self.depths = self.imgs
            self.imgs = ImageStickDataset.load_imgs(self)

    def __getitem__(self, idx):
        _, labels = super().__getitem__(idx)

        if self.pre_load:
            imgs = self.imgs[
                idx * self.traj_skip : idx * self.traj_skip + self.traj_len
            ]
            depths = self.depths[
                idx * self.traj_skip : idx * self.traj_skip + self.traj_len
            ]
        else:
            imgs = []
            depths = []
            for key in self.img_keys[
                idx * self.traj_skip : idx * self.traj_skip + self.traj_len
            ]:
                img = Image.open(str(self.img_pth / key))
                img = self.preprocess_img_transforms(img)
                imgs.append(img)
                depth = load_depth(str(self.depth_pth / (key[:-4] + ".depth")))
                depth = cv2.resize(depth, (self.img_size, self.img_size)) / 4
                # add a new axis at the 0th dimension
                depth = depth[None, :, :]
                depth = torch.from_numpy(depth)
                depths.append(depth)
            # add a nex axis at the beginning
            imgs = torch.stack(imgs, dim=0)
            depths = torch.stack(depths, dim=0)

        if self.img_transforms:
            imgs = self.img_transforms(imgs)
        if self.depth_transforms:
            depths = self.depth_transforms(depths)

        if self.concat_img_depth:
            imgs = torch.cat((imgs, depths), dim=1)
            return imgs, labels

        return imgs, depths, labels


def get_train_image_stick_dataset(
    data_path,
    traj_len=1,
    traj_skip=1,
    time_skip=3,
    time_offset=5,
    time_trim=5,
    img_size=224,
    pre_load=True,
    apply_transforms=False,
    val_mask=None,
    mask_texts=None,
    cfg=None,
):
    # add transforms for normalization and converting to float tensor
    if type(data_path) == str:
        data_path = Path(data_path)

    if apply_transforms:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )
    else:
        transforms = None

    train_traj_paths, _, _ = iter_dir_for_traj_pths(data_path, val_mask, mask_texts)
    # train_traj_paths = train_traj_paths[:16]
    # concatenate all the Datasets for all the trajectories
    train_dataset = ConcatDataset(
        [
            ImageStickDataset(
                traj_path,
                traj_len,
                time_skip,
                time_offset,
                time_trim,
                traj_skip,
                img_size,
                pre_load=pre_load,
                transforms=transforms,
            )
            for traj_path in train_traj_paths
        ]
    )

    return train_dataset, None, None


def get_image_stick_dataset(
    data_path,
    traj_len=1,
    traj_skip=1,
    time_skip=4,
    time_offset=5,
    time_trim=5,
    img_size=224,
    pre_load=True,
    apply_transforms=True,
    val_mask=None,
    mask_texts=None,
    cfg=None,
):
    # add transforms for normalization and converting to float tensor
    if type(data_path) == str:
        data_path = Path(data_path)

    if apply_transforms:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )
    else:
        transforms = None

    train_traj_paths, val_traj_paths, test_traj_paths = iter_dir_for_traj_pths(
        data_path, val_mask, mask_texts
    )
    # train_traj_paths = train_traj_paths[:64]
    # val_traj_paths = val_traj_paths[:16]
    # test_traj_paths = test_traj_paths[:16]
    # concatenate all the Datasets for all the trajectories
    train_dataset = ConcatDataset(
        [
            ImageStickDataset(
                traj_path,
                traj_len,
                time_skip,
                time_offset_n,
                time_trim,
                traj_skip,
                img_size,
                pre_load=pre_load,
                transforms=transforms,
            )
            for traj_path, time_offset_n in itertools.product(
                train_traj_paths, [time_offset, time_offset + 2]
            )
        ]
    )

    act_mean, act_std = get_act_mean_std(train_dataset)
    act_metrics = {"mean": act_mean, "std": act_std}
    if cfg is not None:
        cfg["dataset"]["act_metrics"] = act_metrics

    for dataset in train_dataset.datasets:
        dataset.set_act_metrics(act_metrics)

    if len(val_traj_paths) > 0:
        val_dataset = ConcatDataset(
            [
                ImageStickDataset(
                    traj_path,
                    traj_len,
                    time_skip,
                    time_offset,
                    time_trim,
                    traj_skip,
                    img_size,
                    pre_load=pre_load,
                    transforms=transforms,
                )
                for traj_path in val_traj_paths
            ]
        )
        for dataset in val_dataset.datasets:
            dataset.set_act_metrics(act_metrics)
    else:
        val_dataset = None

    if len(test_traj_paths) > 0:
        test_dataset = ConcatDataset(
            [
                ImageStickDataset(
                    traj_path,
                    traj_len,
                    time_skip,
                    time_offset,
                    time_trim,
                    traj_skip,
                    img_size,
                    pre_load=pre_load,
                    transforms=transforms,
                )
                for traj_path in test_traj_paths
            ]
        )
        for dataset in test_dataset.datasets:
            dataset.set_act_metrics(act_metrics)
    else:
        test_dataset = None

    return train_dataset, val_dataset, test_dataset
