import hydra

from omegaconf import DictConfig

from clip_policy.clip import *
from clip_policy.datasets import *
from clip_policy.models import *
from clip_policy.utils import *
from clip_policy.robot import *

import sys


def save_state_variables(cfg):
    # Get the query vocab
    query = cfg.query
    words = extract_from_query(query)
    # Get the embeddings from the received query
    text_emb = TextEmbeddings()
    query_clip_emb = text_emb.get(words)

    # Sort the datasets
    data_path = Path(cfg.data_path)
    all_traj_paths, _, _ = iter_dir_for_traj_pths(data_path, None, None)
    costs = []
    used_traj_paths, used_offsets = [], []
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ]
    )
    for traj_path, time_offset_n in itertools.product(all_traj_paths, [12, 14]):
        dataset = ImageStickDataset(
            traj_path=traj_path,
            traj_len=1,
            traj_skip=1,
            time_skip=4,
            time_offset=time_offset_n,
            time_trim=5,
            img_size=224,
            pre_load=True,
            transforms=transforms,
        )

        costs.append(dataset.get_cost_from_clip(query_clip_emb.T))
        used_traj_paths.append(traj_path)
        used_offsets.append(time_offset_n)

    sorted_traj_idx = np.argsort(costs)
    selected_traj_paths = np.array(used_traj_paths)[sorted_traj_idx][
        : cfg.max_dataset_num
    ]
    selected_time_offsets = np.array(used_offsets)[sorted_traj_idx][
        : cfg.max_dataset_num
    ]

    dataset = ConcatDataset(  # This will be the final dataset
        [
            ImageStickDataset(
                traj_path=traj_path,
                traj_len=1,
                traj_skip=1,
                time_skip=4,
                time_offset=time_offset_n,
                time_trim=5,
                img_size=224,
                pre_load=True,
                transforms=transforms,
            )
            for traj_path, time_offset_n in zip(
                selected_traj_paths, selected_time_offsets
            )
        ]
    )

    print("Selected traj paths: {}".format(selected_traj_paths))

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Get the VINN action in a loop with the given dataloaders
    # Initialize the VINN model
    encoder = ImageEncoder(encoder_type="resnet18", pretrained=True)
    act_mean, act_std = get_act_mean_std(dataset)
    act_metrics = {"mean": act_mean, "std": act_std}
    vinn_cfg = {"dataset": {}}
    vinn_cfg["dataset"]["act_metrics"] = act_metrics
    vinn_cfg["buffer_size"] = cfg.buffer_size
    model = VINN(encoder=encoder, cfg=vinn_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.k = cfg.k
    model.bs = cfg.batch_size
    model.set_dataset(dataloader)
    model.to(device)
    model.eval()
    model.save_state_variables("state_variables.pkl")

    return model


@hydra.main(version_base=None, config_path="clip_policy/configs", config_name="policy")
def main(cfg: DictConfig) -> None:
    if cfg.save_state_variables:
        model = save_state_variables(cfg)
    else:
        # Load the model
        # Initialize the VINN model
        encoder = ImageEncoder(encoder_type="resnet18", pretrained=True)
        vinn_cfg = {"dataset": {}, "action_space": cfg.action_space}
        vinn_cfg["buffer_size"] = cfg.buffer_size
        model = VINN(encoder=encoder, cfg=vinn_cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.k = cfg.k
        model.bs = cfg.batch_size
        model.to(device)
        model.eval()
        model.load_state_variables("state_variables.pkl")

    # Start and run the controller
    controller = Controller(cfg)
    controller.setup_model(model)
    controller.run()


if __name__ == "__main__":
    main()
