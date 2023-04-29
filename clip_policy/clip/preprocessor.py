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
# sys.path.insert(0, "/scratch/ar7420/VINN/Detic/third_party/CenterNet2/")
# sys.path.insert(0, "/scratch/ar7420/VINN/Detic/")
# sys.path.insert(0,'/home/irmak/Workspace/clip-policy/third_party/CenterNet2/') # NOTE: Make sure this will work
from centernet.config import add_centernet_config

from detic.config import add_detic_config
from detic.modeling.text.text_encoder import build_text_encoder
from detic.modeling.utils import reset_cls_test

from clip_policy.utils import *
from .embeddings import TextEmbeddings


class Preprocessor:
    def __init__(
        self,
        env_path,  # Main directory
        score_thresh_test,
        classes,
        visualize=True,
        cfg_merge_path="/home/irmak/Workspace/clip-policy/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        model_weights_path="https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
        # cfg_merge_path="/scratch/ar7420/VINN/clip-policy/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        # model_weights_path="/scratch/ar7420/VINN/clip-policy/weights/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
        zeroshot_weight_path="rand",
        one_class_per_proposal=True,
    ):
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(cfg_merge_path)
        cfg.MODEL.WEIGHTS = model_weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
            score_thresh_test  # set threshold for this model
        )
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = zeroshot_weight_path
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = one_class_per_proposal  # For better visualization purpose. Set to False for all classes.
        self.predictor = DefaultPredictor(cfg)

        # Build the text encoder - used in getting the clip embeddings
        self.text_embeddings = TextEmbeddings()

        # Get the major classifier
        vocabulary = "custom"  # change to 'lvis', 'objects365', 'openimages', or 'coco'
        self.metadata = MetadataCatalog.get("__unused")
        self.metadata.thing_classes = classes
        all_vocab_classifier = self.text_embeddings.get(classes)
        num_classes = len(classes)
        reset_cls_test(self.predictor.model, all_vocab_classifier, num_classes)
        self.classes = classes

        # Create the dump directory
        self.env_path = env_path
        self.roots = sorted(glob.glob(os.path.join(env_path, "*")))
        if visualize:  # If visualize create the directory to dump images
            for root in self.roots:
                os.makedirs(os.path.join(root, "detected_images"), exist_ok=True)
        self.visualize = visualize

    def _get_image(self, path):
        return cv2.imread(path)

    def _get_class_names(self, pred_class_idx):
        class_names = []
        for i in pred_class_idx:
            class_names.append(self.classes[int(i)])

        return class_names

    def process_image(self, path):
        img = self._get_image(path)
        outputs = self.predictor(img)

        # Visualize if wanted
        if self.visualize:
            self.visualize_detection(img, outputs, path)

        # Get the text embeddings for each class that is predicted
        fields = outputs["instances"].get_fields()
        pred_class_idx = fields["pred_classes"]
        pred_classes = self._get_class_names(
            pred_class_idx
        )  # Names of the predicted classes

        text_embeddings = self.text_embeddings.get(pred_classes)
        scores = fields["scores"]

        label = dict(
            pred_classes=pred_classes,
            clip_embeddings=text_embeddings.numpy(),
            scores=scores.detach().cpu().numpy(),
        )

        return label

    def process_env(self):  # Path of the root
        for root in self.roots:
            image_paths = sorted(glob.glob(os.path.join(root, "images/*")))

            # Create the labels directory
            # We will create one big labels file
            root_labels = dict()  # Dump this as a json file
            pbar = tqdm(total=len(image_paths))

            for image_path in image_paths:
                # print(f'image_path: {image_path}')
                image_label = self.process_image(image_path)
                root_labels[image_path] = image_label

                pbar.update(1)
                pbar.set_description(f"Processed Image: {image_path}")

            # Dump this as a json file to the root
            with open(os.path.join(root, "detections.pkl"), "wb") as outfile:
                pickle.dump(root_labels, outfile)

    def visualize_detection(self, img, outputs, img_path):
        v = Visualizer(img[:, :, ::-1], self.metadata)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Set the dump img path
        dump_img_path_list = img_path.split("/")
        dump_img_path_list[-2] = "detected_images"
        dump_img_path = "/".join(dump_img_path_list)

        # Dump the new detected image
        cv2.imwrite(dump_img_path, out.get_image()[:, :, ::-1])
