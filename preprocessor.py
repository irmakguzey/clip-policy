# Lets say we have a preprocessor module that receives the dataloaders
# Traverses whole data directory
# Gets the images
# Initializes the Detic class
# And get all the embeddings

# Some basic setup:
# Setup detectron2 logger
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

from clip_policy.utils import *

class Preprocessor:
    def __init__(
            self,
            env_path, # Main directory
            cfg_merge_path = '/home/irmak/Workspace/clip-policy/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml',
            model_weights_path = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth',
            score_thresh_test = 0.5,
            zeroshot_weight_path = 'rand',
            one_class_per_proposal = True,
            classes = ['cabinet', 'drawer', 'gripper']
    ):

        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(cfg_merge_path)
        cfg.MODEL.WEIGHTS = model_weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh_test  # set threshold for this model
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = zeroshot_weight_path
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = one_class_per_proposal # For better visualization purpose. Set to False for all classes.
        self.predictor = DefaultPredictor(cfg)

        # Build the text encoder - used in getting the clip embeddings
        self.text_encoder = build_text_encoder(pretrain=True)
        self.text_encoder.eval()

        # Get the major classifier
        vocabulary = 'custom' # change to 'lvis', 'objects365', 'openimages', or 'coco'
        metadata = MetadataCatalog.get('__unused')
        metadata.thing_classes = classes
        all_vocab_classifier = self._get_clip_embeddings(classes)
        num_classes = len(classes)
        reset_cls_test(self.predictor.model, all_vocab_classifier, num_classes)
        self.classes = classes

        # Set the image transform
        self.image_transform = T.Compose([
            T.Resize((480,480)),
            # T.Lambda(self._crop_transform), - TODO ask and then do this!
            T.ToTensor(),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS),
        ])

        # Create the dump directory
        self.env_path = env_path 
        self.roots = glob.glob(os.path.join(env_path,'*'))


    # def _crop_transform(self, image):
    #     # Vision transforms used
    #     return crop(image, 0,0,480,480)

    def _get_clip_embeddings(self, vocabulary, prompt='a '):
        texts = [prompt + x for x in vocabulary]
        emb = self.text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
        return emb
    
    def _get_image(self, path):
        return cv2.imread(path)
    
    def _get_class_names(self, pred_class_idx):
        class_names = []
        for i in pred_class_idx:
            class_names.append(self.classes[i])

        return class_names

    def process_image(self, path):
        img = self._get_image(path)
        # plt.imshow(img)
        # print(img.shape)
        outputs = self.predictor(img) 
        
        # Get the text embeddings for each class that is predicted
        fields = outputs['instances'].get_fields()
        pred_class_idx = fields['pred_classes']
        pred_classes = self._get_class_names(pred_class_idx) # Names of the predicted classes

        text_embeddings = self._get_clip_embeddings(pred_classes)
        scores = fields['scores']

        label = dict(
            pred_classes = pred_classes,
            clip_embeddings = text_embeddings.numpy(), 
            scores = scores.detach().cpu().numpy()
        )

        return label

    def process_env(self): # Path of the root

        for root in self.roots:
            image_paths = sorted(glob.glob(os.path.join(root, 'images/*')))

            # Create the labels directory 
            # os.makedirs(os.path.join(root, 'labels'), exist_ok=True) 
            # We will create one big labels file
            root_labels = dict() # Dump this as a json file
            pbar = tqdm(total=len(image_paths))
            
            for image_path in image_paths:
                print(f'image_path: {image_path}')
                image_label = self.process_image(image_path)
                root_labels[image_path] = image_label 

                pbar.update(1)
                pbar.set_description(f'Processed Image: {image_path}')

            # Dump this as a json file to the root
            with open(os.path.join(root, 'labels.pkl'), "wb") as outfile:
                pickle.dump(root_labels, outfile)

if __name__ == '__main__':
    env_path = '/home/irmak/Workspace/clip-policy/data'
    preprocessor = Preprocessor(
        env_path = env_path
    )
    print(f'preprocessor.root: {preprocessor.roots}')

    preprocessor.process_env()





