import torch
import torch.nn as nn
import torch.nn.functional as F
from byol_pytorch import BYOL
from torchvision import transforms as T
import os
import wandb
from tqdm import tqdm
from trainers.trainer import Trainer
from utils.image_plots import overlay_action, display_image_in_grid
import cv2


class VINNTrainer(Trainer):
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        experiment,
        save_every,
        cfg=None,
        max_k=20,
    ):
        super().__init__(
            model, optimizer, scheduler, device, experiment, save_every, cfg
        )
        self.loss_fn = nn.MSELoss()
        self.max_k = max_k
        self.model.eval()

    def train(self, train_dataloader, val_dataloader, test_dataloader, epochs):

        self.model.to(self.device)
        self.model.set_dataset(train_dataloader)
        for k in tqdm(range(1, self.max_k + 1)):
            val_loss = 0
            translation_val_loss = 0
            rotation_val_loss = 0
            gripper_val_loss = 0

            for i, (x, y) in enumerate(val_dataloader):
                x = x.to(self.device)
                x = x.squeeze()
                y = y.to(self.device)
                y = y.squeeze()

                y_hat = self.model(x, k)

                loss = self.loss_fn(y_hat, y)
                translation_val_loss += self.loss_fn(y_hat[:, :3], y[:, :3]).item()
                rotation_val_loss += self.loss_fn(y_hat[:, 3:6], y[:, 3:6]).item()
                gripper_val_loss += self.loss_fn(y_hat[:, 6:], y[:, 6:]).item()

                val_loss += loss.item()
            val_loss /= len(val_dataloader)
            translation_val_loss /= len(val_dataloader)
            rotation_val_loss /= len(val_dataloader)
            gripper_val_loss /= len(val_dataloader)

            wandb.log({"val_loss": val_loss}, step=k)
            wandb.log({"translation_val_loss": translation_val_loss}, step=k)
            wandb.log({"rotation_val_loss": rotation_val_loss}, step=k)
            wandb.log({"gripper_val_loss": gripper_val_loss}, step=k)

        if not os.path.exists("./weights"):
            os.makedirs("./weights")

        self.model.save_state_variables(f"./weights/{self.experiment}.pkl")

    def eval_dataset(
        self, dataset, k=4, start=None, end=None, plot_freq=1, vector_scale=14
    ):
        img_grid = []
        label_grid = []
        loss = 0

        for i, (x, y) in enumerate(dataset):
            if start is not None and i < start:
                continue
            if end is not None and i > end:
                break

            query_img = cv2.imread(str(dataset.get_img_pths(i)[0]))

            y = torch.from_numpy(y).to(self.device)
            x = x.to(self.device)

            normalzed_action = y
            action = self.model.denorm_action(normalzed_action)
            action = action.squeeze().detach().cpu().numpy()

            query_img = overlay_action(
                action, query_img, color=(0, 255, 0), vector_scale=vector_scale
            )

            y_hat, indices = self.model(x, k, return_indices=True)

            nromalized_pred_action = y_hat.squeeze().detach()
            pred_action = self.model.denorm_action(nromalized_pred_action)

            l = self.loss_fn(y_hat, y)
            loss += l

            query_img = overlay_action(
                pred_action.cpu().numpy(),
                query_img,
                color=(255, 0, 0),
                vector_scale=vector_scale,
                shift_start_point=True,
            )

            if plot_freq != 0 and i % plot_freq == 0:
                img_grid.append([])
                label_grid.append([])
                img_grid[-1].append(query_img)
                label_grid[-1].append("query, loss {:.3f}".format(l.item()))

                # label_grid[-1].append('query, loss {:.3f}'.format(l.item()) + \
                #     'p: '+str(y_hat.squeeze().detach().cpu().numpy()) + \
                #     'a: '+str(y.squeeze().detach().cpu().numpy()))

                indices = indices[0]
                for j in range(k):
                    img = cv2.imread(str(self.model.img_pths[indices[j]]))
                    nbhr_action = (
                        self.model.denorm_action(
                            self.model.actions[indices[j]].to(self.device)
                        )
                        .squeeze()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    img = overlay_action(
                        nbhr_action, img, color=(0, 255, 0), vector_scale=vector_scale
                    )
                    if plot_freq != 0 and i % plot_freq == 0:
                        img_grid[-1].append(img)
                        label_grid[-1].append("nbhr")

        loss /= len(dataset)
        if plot_freq != 0:
            display_image_in_grid(img_grid, label_grid)
        return loss

    def validate(self, dataloader):
        pass

    def test(self, dataloader):
        pass
