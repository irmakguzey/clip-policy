import torch
import torch.nn as nn
import torch.nn.functional as F
from byol_pytorch import BYOL
from torchvision import transforms as T
import os
import wandb
from tqdm import tqdm
from trainers.trainer import Trainer


class BcTrainer(Trainer):
    def __init__(
        self, model, optimizer, scheduler, device, experiment, save_every, cfg=None
    ):
        super().__init__(
            model, optimizer, scheduler, device, experiment, save_every, cfg
        )
        self.loss_fn = nn.MSELoss()

    def train(self, train_dataloader, val_dataloader, test_dataloader, epochs):
        self.model.to(self.device)
        self.model.train()
        for epoch in tqdm(range(epochs)):
            train_loss = 0
            self.current_epoch = epoch
            for i, (x, y) in enumerate(train_dataloader):
                x = x.to(self.device)
                x = x.squeeze()
                y = y.to(self.device)
                y = y.squeeze().float()

                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_dataloader)
            wandb.log({"train_loss": train_loss}, step=epoch)
            self.validate(val_dataloader)

            if epoch % self.save_every == 0:
                self.save(epoch)

    def validate(self, dataloader):
        self.model.eval()
        val_loss = 0
        translation_val_loss = 0
        rotation_val_loss = 0
        gripper_val_loss = 0
        for i, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            x = x.squeeze()
            y = y.to(self.device)
            y = y.squeeze()

            y_hat = self.model(x)
            loss = self.loss_fn(y_hat, y)
            val_loss += loss.item()
            translation_val_loss += self.loss_fn(y_hat[:, :3], y[:, :3]).item()
            rotation_val_loss += self.loss_fn(y_hat[:, 3:6], y[:, 3:6]).item()
            gripper_val_loss += self.loss_fn(y_hat[:, 6:], y[:, 6:]).item()

        val_loss /= len(dataloader)
        translation_val_loss /= len(dataloader)
        rotation_val_loss /= len(dataloader)
        gripper_val_loss /= len(dataloader)

        wandb.log({"val_loss": val_loss}, step=self.current_epoch)
        wandb.log(
            {"translation_val_loss": translation_val_loss}, step=self.current_epoch
        )
        wandb.log({"rotation_val_loss": rotation_val_loss}, step=self.current_epoch)
        wandb.log({"gripper_val_loss": gripper_val_loss}, step=self.current_epoch)

    def test(self, dataloader):
        pass
