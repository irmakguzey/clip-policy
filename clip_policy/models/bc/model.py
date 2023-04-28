# naive bc model that takes in encoder, actionspace and predicts actions
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCModel(nn.Module):
    def __init__(
        self,
        encoder,
        action_space=None,
        enc_weight_pth=None,
        freeze_encoder=False,
        cfg=None,
    ):
        super(BCModel, self).__init__()
        self.encoder = encoder
        self.cfg = cfg

        if enc_weight_pth is not None:
            self.encoder.load_state_dict(torch.load(enc_weight_pth))

        if cfg is not None:
            img_size = cfg["dataset"]["img_size"]
        encoder_out_dim = self.encoder(
            torch.zeros(1, 3, img_size[0], img_size[1])
        ).shape[1]

        self.action_space = action_space
        self.fc1 = nn.Linear(encoder_out_dim, 512)
        self.fc2 = nn.Linear(512, self.action_space)

        self.set_act_metrics(cfg)

        if freeze_encoder:
            self.encoder.freeze()

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

    def get_action(self, img):
        self.eval()
        with torch.no_grad():
            act = self.forward(img).squeeze().detach()
            act = self.denorm_action(act)
        # print(act.shape)
        return act.cpu()

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
