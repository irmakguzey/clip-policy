# VINN model that uses encoder and nearest neighbor to predict action
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pickle as pkl


class VINN(nn.Module):
    def __init__(self, encoder, enc_weight_pth=None, cfg=None):
        super(VINN, self).__init__()
        self.encoder = encoder
        self.cfg = cfg

        self.k = 5

        self.set_act_metrics(cfg)

        self.representations = None
        self.actions = None
        self.img_pths = None
        self.imgs = None
        softmax = nn.Softmax(dim=1)
        self.dist_scale_func = lambda x: (softmax(-x))
        self.encoder.eval()
        self.device = "cpu"

    def set_act_metrics(self, cfg):
        if cfg is not None and "act_metrics" in cfg["dataset"]:
            act_metrics = cfg["dataset"]["act_metrics"]
            self.act_mean = nn.Parameter(act_metrics["mean"].float())
            self.act_std = nn.Parameter(act_metrics["std"].float())
        else:
            self.act_mean = nn.Parameter(torch.zeros(self.action_space))
            self.act_std = nn.Parameter(torch.ones(self.action_space))

        self.act_mean.requires_grad = False
        self.act_std.requires_grad = False

    def denorm_action(self, action):
        return action * self.act_std + self.act_mean

    def to(self, device):
        super().to(device)
        self.device = device
        self.encoder.to(device)

    def set_dataset(self, dataloader):
        self.bs = dataloader.batch_size
        for dataset in tqdm(dataloader.dataset.datasets): # Traverse through multiple datasets
            for i, (image, label) in enumerate(dataset):
                image = image.to(self.device)
                label = torch.Tensor(label).to("cpu").detach()
                pth = dataset.get_img_pths(i)[0]
                representation = self.encoder(image).to("cpu").detach()
                if self.representations is None: # If this is the first batch
                    self.representations = representation
                    self.actions = label
                    self.img_pths = [pth]
                    self.imgs = [image.to("cpu").detach().numpy()]
                else: # For all the rest batches
                    self.representations = torch.cat(
                        (self.representations, representation), 0
                    )
                    self.actions = torch.cat((self.actions, label), 0)
                    self.img_pths.append(pth)
                    self.imgs.append(image.to("cpu").detach().numpy())

    def get_action(self, img, return_indices=False):
        self.encoder.eval()
        with torch.no_grad():
            if return_indices:
                act, indices = self(img, return_indices=True)
            else:
                act = self(img).squeeze().detach().cpu()
            act = self.denorm_action(act)

        return act if not return_indices else (act, indices)

    def __call__(self, batch_images, k=None, return_indices=False):
        if k is None:
            k = self.k

        all_distances = torch.zeros(
            (batch_images.shape[0], self.representations.shape[0])
        ) 

        for i in range(0, self.representations.shape[0] // self.bs + 1):

            dat_rep = self.representations[
                i * self.bs : min((i + 1) * self.bs, self.representations.shape[0])
            ].to(self.device)
            batch_rep = self.encoder(batch_images).to(self.device)
            all_distances[
                :, i * self.bs : min((i + 1) * self.bs, self.representations.shape[0])
            ] = (torch.cdist(batch_rep, dat_rep).to("cpu").detach()) # Getting the distance and saving it as such here

        top_k_distances, indices = torch.topk(all_distances, k, dim=1, largest=False)
        top_k_actions = self.actions[indices].to(self.device)
        weights = self.dist_scale_func(top_k_distances.to(self.device))
        pred = torch.sum(top_k_actions * weights.unsqueeze(-1), dim=1)

        if return_indices:
            return pred, indices

        return pred

    def save_state_variables(self, save_path):
        save_vars = [
            self.representations,
            self.actions,
            self.act_mean,
            self.act_std,
            self.img_pths,
            self.imgs
        ]
        save_var_names = [
            "representations",
            "actions",
            "act_mean",
            "act_std",
            "img_pths",
            "imgs",
        ]
        save_dict = {}
        for i, var in enumerate(save_vars):
            save_dict[save_var_names[i]] = var
        pkl.dump(save_dict, open(save_path, "wb"))

        # also save encder weights with same name but .pth extension
        torch.save(self.encoder.state_dict(), save_path[:-4] + ".pth")

    def load_state_variables(self, load_path):
        print("Loading state variables from {}".format(load_path))
        load_dict = pkl.load(open(load_path, "rb"))
        self.representations = load_dict["representations"]
        self.actions = load_dict["actions"]
        self.act_mean = load_dict["act_mean"]
        self.act_std = load_dict["act_std"]
        self.img_pths = load_dict["img_pths"]
        self.imgs = load_dict["imgs"]

        self.encoder.load_state_dict(torch.load(load_path[:-4] + ".pth"))
        self.encoder.eval()
