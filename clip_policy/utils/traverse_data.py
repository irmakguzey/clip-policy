import random
import os
import numpy as np
from pathlib import Path


def mask(mask_words, path):
    for mask_word in mask_words:
        if mask_word in str(path):
            return True
    return False


def iter_dir(data_path, val_mask=None, mask_texts=None, depth=0):
    # print('Iterating through: ', data_path)
    if depth == 3:
        return [data_path]

    traj_paths = []
    for child_pth in data_path.iterdir():
        if depth == 0 and (
            (val_mask is not None and mask(val_mask.get("home", None), child_pth))
            or not child_pth.is_dir()
        ):
            continue
        elif depth == 1 and (
            (val_mask is not None and mask(val_mask.get("env", None), child_pth))
            or not child_pth.is_dir()
        ):
            continue
        elif depth == 2 and (
            (val_mask is not None and mask(val_mask.get("traj", None), child_pth))
            or not child_pth.is_dir()
        ):
            continue

        if mask_texts is not None and mask(mask_texts, child_pth):
            continue

        if depth < 2:
            traj_paths.extend(iter_dir(child_pth, val_mask, mask_texts, depth + 1))
        else:
            traj_paths.append(child_pth)

    return traj_paths


def iter_dir_val(data_path, val_mask, depth=0):
    print("Iterating through: ", data_path)
    if depth == 3:
        return [data_path]

    traj_paths = []
    for child_pth in data_path.iterdir():
        if val_mask is None or not child_pth.is_dir():
            break

        if depth < 2:
            traj_paths.extend(iter_dir_val(child_pth, val_mask, depth + 1))
        else:
            # print('child_pth: ', child_pth)
            # print(mask(val_mask['env'], data_path.parent))
            # print(val_mask['env'], data_path.parent)
            if (
                mask(val_mask["traj"], child_pth)
                or mask(val_mask["env"], data_path)
                or mask(val_mask["home"], data_path.parent)
            ):
                traj_paths.append(child_pth)

    return traj_paths


def iter_dir_for_traj_pths(
    data_path,
    val_mask=None,
    mask_texts=None,
    split_test_from_val=False,
    splt_percent=0.0,
):
    base_name = data_path.name
    if "Home" in str(base_name):
        train_traj_paths = iter_dir(data_path, val_mask, mask_texts, depth=1)
        val_traj_paths = iter_dir_val(data_path, val_mask, depth=1)
    elif "Env" in str(base_name) or 'data' in str(base_name):
        train_traj_paths = iter_dir(data_path, val_mask, mask_texts, depth=2)
        val_traj_paths = iter_dir_val(data_path, val_mask, depth=2)
    else:
        train_traj_paths = iter_dir(data_path, val_mask, mask_texts)
        val_traj_paths = iter_dir_val(data_path, val_mask)

    print("Total number of trajectories: ", len(train_traj_paths))

    test_traj_paths = []

    if split_test_from_val:
        n = int(len(val_traj_paths) * splt_percent)
        to_delete = set(random.sample(range(len(val_traj_paths)), n))
        test_traj_paths = [x for i, x in enumerate(val_traj_paths) if i in to_delete]
        val_traj_paths = [x for i, x in enumerate(val_traj_paths) if i not in to_delete]
    else:
        n = int(len(train_traj_paths) * splt_percent)
        to_delete = set(random.sample(range(len(train_traj_paths)), n))
        test_traj_paths = [x for i, x in enumerate(train_traj_paths) if i in to_delete]
        train_traj_paths = [
            x for i, x in enumerate(train_traj_paths) if i not in to_delete
        ]

    # print('val_traj_pths: ', val_traj_paths)
    # print('train_traj_paths: ', train_traj_paths)
    # print('test_traj_paths: ', test_traj_paths)
    print("total number of train trajectories: ", len(train_traj_paths))
    print("total number of test trajectories: ", len(test_traj_paths))
    print("Total number of val trajectories: ", len(val_traj_paths))

    # return train_traj_paths[:32], val_pth, test_traj_paths
    return train_traj_paths, val_traj_paths, test_traj_paths
