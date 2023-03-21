from dataset import create_detectron_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import cv2
import random
import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import argparse
import sys
from utils import *


def main(config, wandb_name):
    # Create Predictor
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config["model_path"]))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config["model_path"])
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # confidence threshold
    cfg.MODEL.DEVICE = "cuda"
    predictor = DefaultPredictor(cfg)

    if not config["bool_evaluate"]:
        # visualize KITTI-MOTS ground truth and predictions
        dataset_dicts = create_detectron_dataset(config["dataset_path"])

        random.seed(42)
        for sample in random.sample(dataset_dicts, 10):
            img = cv2.imread(sample["file_name"])

            # Ground truth
            visualizer = Visualizer(img[:, :, ::-1],MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
            out = visualizer.draw_dataset_dict(sample)
            plt.figure(figsize=(15, 7))
            plt.imshow(out.get_image()[:, :, ::-1][..., ::-1])
            plt.title("Ground truth")
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            plt.tight_layout()
            plt.show()

            # Predictions
            predictions = predictor(img)
            visualizer = Visualizer(img[:, :, ::-1],MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
            out = visualizer.draw_instance_predictions(predictions["instances"].to("cpu"))
            plt.figure(figsize=(15, 7))
            plt.imshow(out.get_image()[:, :, ::-1][..., ::-1])
            plt.title("Predicted")
            plt.tight_layout()
            plt.show()

    # if config["bool_evaluate"]:
    #     # Evaluate
    #     evaluator = COCOEvaluator("kitti_val", cfg, False, output_dir="./output")
    #     val_loader = build_detection_test_loader(cfg, "kitti_val")
    #     print(inference_on_dataset(predictor.model, val_loader, evaluator))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/inference_and_eval.yaml")
    parser.add_argument("--wandb_name", default=None)
    args = parser.parse_args(sys.argv[1:])

    config_path = args.config
    wandb_name = args.wandb_name

    config = load_yaml_config(config_path)

    main(config, wandb_name)
