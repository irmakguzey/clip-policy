import hydra

from omegaconf import DictConfig

from clip_policy.clip import *
from clip_policy.datasets import *
from clip_policy.models import *
from clip_policy.utils import *
from clip_policy.robot import *

@hydra.main(version_base=None, config_path='clip_policy/configs', config_name='policy')
def main(cfg : DictConfig) -> None:
    
    # Get the query vocab
    query = cfg.query
    words = extract_from_query(query)
    # Get the embeddings from the received query
    text_emb = TextEmbeddings()
    query_clip_emb = text_emb.get(words)

    # Sort the datasets
    data_path = Path(cfg.data_path)
    all_traj_paths, _, _ = iter_dir(data_path, None, None)
    costs = []
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])
    for traj_path in all_traj_paths:
        dataset = ImageStickDataset(
            traj_path = traj_path,
            traj_len = 1,
            traj_skip = 1,
            time_skip = 4,
            time_offset = 5,
            time_trim = 5,
            img_size = 224,
            pre_load = True,
            transforms = transforms,
        )

        costs.append(
            dataset.get_cost_from_clip(query_clip_emb.T)
        )
    sorted_traj_idx = np.argsort(costs)
    selected_traj_paths = np.array(all_traj_paths)[sorted_traj_idx][:cfg.max_dataset_num]
    dataset = ConcatDataset( # This will be the final dataset
        [
            ImageStickDataset(
                traj_path = traj_path,
                traj_len = 1,
                traj_skip = 1,
                time_skip = 4,
                time_offset = 5,
                time_trim = 5,
                img_size = 224,
                pre_load = True,
                transforms = transforms,
            )
            for traj_path in selected_traj_paths
        ]
    )
    dataloader = DataLoader(
        dataset = dataset, 
        batch_size = cfg.batch_size,
        shuffle=False,
        num_workers = 4,
        pin_memory = True
    )

    # Get the VINN action in a loop with the given dataloaders
    # Initialize the VINN model
    encoder = ImageEncoder(
        encoder_type = 'resnet18',
        pretrained = True
    )
    act_mean, act_std = get_act_mean_std(dataset)
    act_metrics = {"mean": act_mean, "std": act_std}
    vinn_cfg = {'dataset': {}}
    vinn_cfg['dataset']['act_metrics'] = act_metrics
    model = VINN(
        encoder = encoder,
        cfg = cfg
    )
    model.k = cfg.k
    model.bs = cfg.batch_size
    model.set_dataset(dataset)
    model.eval()

    # Start and run the controller
    controller = Controller(cfg)
    controller.setup_model(model)
    controller.run()

if __name__ == '__main__':
    main()