import hydra
import torch

from pathlib import Path
from utils import set_seed_everywhere
from torch.utils.data import DataLoader

from torchlars import LARS

import wandb
from omegaconf import OmegaConf



class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.work_dir = Path.cwd()
        print("Saving to", self.work_dir)

        set_seed_everywhere(cfg.seed)
        self.device = cfg.device
        dict_cfg = OmegaConf.to_container(self.cfg, resolve=True)

        self.train_set, self.val_set, self.test_set = hydra.utils.instantiate(
            cfg.dataset
        )(cfg=dict_cfg)
        self._setup_loaders()
        self.wandb_run = wandb.init(
            dir=self.work_dir,
            project=cfg.experiment,
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )
        wandb.config.update(
            {
                "save_path": self.work_dir,
            }
        )

        self.model = hydra.utils.instantiate(cfg.model)(cfg=dict_cfg)

        if cfg.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=cfg.lr, momentum=cfg.momentum
            )
        elif cfg.optimizer == "lars":
            self.optimizer = LARS(
                torch.optim.SGD(
                    self.model.parameters(),
                    lr=cfg.lr,
                    momentum=cfg.momentum,
                    weight_decay=cfg.weight_decay,
                )
            )
        else:
            raise NotImplementedError

        # print(dict_cfg['dataset']['img_size'])
        self.trainer = hydra.utils.instantiate(cfg.trainer)(
            model=self.model,
            optimizer=self.optimizer,
            experiment=cfg.experiment,
            device=cfg.device,
            cfg=dict_cfg,
        )

    def _setup_loaders(self):
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.cfg.bs,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        if self.val_set is not None:
            self.val_loader = DataLoader(
                self.val_set,
                batch_size=self.cfg.bs,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
            )
        else:
            self.val_loader = None

        if self.test_set is not None:
            self.test_loader = DataLoader(
                self.test_set,
                batch_size=self.cfg.bs,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
            )
        else:
            self.test_loader = None

    def train(self):
        self.trainer.train(
            self.train_loader, self.val_loader, self.test_loader, self.cfg.epochs
        )


@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    workspace = Workspace(cfg)
    workspace.train()


if __name__ == "__main__":
    main()
