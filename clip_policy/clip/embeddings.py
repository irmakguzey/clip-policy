import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import glob
import sys
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import pickle
import torch
import torchvision.transforms as T 

from tqdm import tqdm 
from torchvision.datasets.folder import default_loader as loader 
from torch.utils import data

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries
sys.path.insert(0,'/home/irmak/Workspace/clip-policy/third_party/CenterNet2/') # NOTE: Make sure this will work 
from centernet.config import add_centernet_config

from detic.config import add_detic_config
from detic.modeling.text.text_encoder import build_text_encoder
from detic.modeling.utils import reset_cls_test

class TextEmbeddings:
    def __init__(self):

        # Build the text encoder - used in getting the clip embeddings
        self.text_encoder = build_text_encoder(pretrain=True)
        self.text_encoder.eval()

    def get(self, vocabulary, prompt='a '):
        texts = [prompt + x for x in vocabulary]
        emb = self.text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
        return emb