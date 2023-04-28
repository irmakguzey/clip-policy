# trainer interface for model training
import os
from abc import ABC, abstractmethod
import torch


class Trainer(ABC):
    def __init__(
        self, model, optimizer, scheduler, device, experiment, save_every, cfg
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.experiment = experiment
        self.save_every = save_every
        self.cfg = cfg
    


    @abstractmethod
    def train(self, train_dataloader, val_dataloader, test_dataloader, epochs):
        pass

    @abstractmethod
    def validate(self, dataloader):
        pass

    @abstractmethod
    def test(self, dataloader):
        pass

    def save(self, epoch):
        if not os.path.exists("./weights"):
            os.makedirs("./weights")
        torch.save(self.model.state_dict(), f"./weights/{self.experiment}_{epoch}.pth")

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
