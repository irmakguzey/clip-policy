import hydra
import torch
from omegaconf import OmegaConf

from clip_policy.clip import *
from clip_policy.datasets import *
from clip_policy.models import *
from clip_policy.utils import *
from clip_policy.robot import *

def get_selected_traj_paths(cfg, query, data_path):
    # Get the query vocab
    words = extract_from_query(query)
    # Get the embeddings from the received query
    text_emb = TextEmbeddings()
    query_clip_emb = text_emb.get(words)

    # Sort the datasets
    data_path = Path(data_path)
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
    for traj_path, time_offset_n in itertools.product(all_traj_paths, [5, 7]):
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

    print('selected_traj_paths: {}, selected_time_offsets: {}'.format(
        selected_traj_paths, selected_time_offsets
    ))

    return selected_traj_paths, selected_time_offsets

@hydra.main(config_path="clip_policy/configs", config_name="run")
def main(cfg):

    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    controller = Controller(cfg = dict_cfg)
    
    model = hydra.utils.instantiate(cfg.model)()
    # model.k = cfg["k"]
    # model.bs = cfg["batch_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.save_state_variables:
        # Set dataset
        data_path = Path(cfg.data_path)
        if cfg.language_conditioning:
            traj_paths, time_offsets = get_selected_traj_paths(cfg, cfg.query, data_path)
        else:
            traj_paths, _, _ = iter_dir_for_traj_pths(data_path, None, None)
            time_offsets = None
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )
        if time_offsets is None:
            dataset = ConcatDataset(
                [
                    ImageStickDataset(
                        traj_path=traj_path,
                        traj_len=1,
                        time_skip=4,
                        time_offset=time_offset_n,
                        time_trim=5,
                        traj_skip=4,
                        img_size=224,
                        pre_load=True,
                        transforms=transforms,
                    )
                    for traj_path, time_offset_n in itertools.product(
                        traj_paths, [5, 7]
                    )
                ]
            )
        else:
            dataset = ConcatDataset(
                [
                    ImageStickDataset(
                        traj_path=traj_path,
                        traj_len=1,
                        time_skip=4,
                        time_offset=time_offset_n,
                        time_trim=5,
                        traj_skip=4,
                        img_size=224,
                        pre_load=True,
                        transforms=transforms,
                    )
                    for traj_path, time_offset_n in zip(
                        traj_paths, time_offsets
                    )
                ]
            )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        model.set_dataset(dataloader)
        model.to(device)
        model.eval()
        model.save_state_variables(cfg['model_pth'])
        # Save state variables
    else:
        model.load_state_variables(cfg["model_pth"])
    
    controller.setup_model(model)
    controller.run()

if __name__ == "__main__":

    main()