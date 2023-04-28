import torch
import torch.nn as nn
import torch.nn.functional as F
from byol_pytorch import BYOL
from torchvision import transforms as T
import os
import accelerate
from utils import setup_accelerate
from tqdm import tqdm
from trainers.trainer import Trainer

import hydra


class BYOLTrainer(Trainer):
    def __init__(
        self, model, optimizer, scheduler, device, experiment, save_every, cfg
    ):
        super().__init__(
            model, optimizer, scheduler, device, experiment, save_every, cfg
        )
        augmentation1 = T.Compose(
            [
                T.RandomResizedCrop(self.cfg["dataset"]["img_size"], scale=(0.6, 1.0)),
                T.RandomApply(
                    torch.nn.ModuleList([T.ColorJitter(0.8, 0.8, 0.8, 0.2)]), p=0.3
                ),
                T.RandomGrayscale(p=0.2),
                T.RandomApply(
                    torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        augmentation2 = None
        self.learner = BYOL(
            self.model,
            image_size=self.cfg["dataset"]["img_size"][0],
            hidden_layer=-1,
            augment_fn=augmentation1,
            augment_fn2=augmentation2,
        )

    def train(self, train_dataloader, val_dataloader, test_dataloader, epochs):
        # train_dataloader = accelerate.Accelerator(train_dataloader)
        self.accelerator, _ = setup_accelerate(
            self.experiment, epochs, "BYOL", self.save_every
        )

        (
            self.model,
            self.learner,
            self.optimizer,
            self.train_dataloader,
        ) = self.accelerator.prepare(
            self.model, self.learner, self.optimizer, train_dataloader
        )

        self.accelerator.register_for_checkpointing(
            self.model, self.optimizer, self.learner
        )
        self.device = self.accelerator.device
        self.learner.to(self.device)
        self.model.to(self.device)
        if not os.path.exists("./weights"):
            os.makedirs("./weights")

        for epoch in tqdm(range(epochs)):
            epoch_loss = 0
            for i, (x, _) in enumerate(train_dataloader):
                x = x.to(self.device)
                x = x.squeeze()
                loss = self.learner(x)
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.accelerator.unwrap_model(self.learner).update_moving_average()

                epoch_loss += loss.item()

            self.accelerator.log(
                {"loss": epoch_loss / len(train_dataloader)}, step=epoch
            )
            if epoch % self.save_every == 0:
                torch.save(
                    self.accelerator.unwrap_model(self.model).state_dict(),
                    f"./weights/{self.experiment}_{epoch}.pth",
                )
                self.accelerator.wait_for_everyone()
                # accelerator.save_state('checkpoint.pth', model, optimizer, lr_scheduler)

    def validate(self, dataloader):
        pass

    def test(self, dataloader):
        pass
