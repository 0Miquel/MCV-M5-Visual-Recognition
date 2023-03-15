from dataset import get_kitti_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import cv2
import random
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
import os
import wandb
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import argparse
import sys
from utils import *


def main(config, wandb_name):
    # REGISTER DATASETS
    for phase in ["train", "val"]:
        DatasetCatalog.register("kitti_" + phase, lambda phase=phase: get_kitti_dataset(config["dataset_path"], phase))
        MetadataCatalog.get("kitti_" + phase).set(thing_classes=['Car', 'Pedestrian'])

    # TRAIN
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config["model_path"]))
    cfg.DATASETS.TRAIN = ("kitti_train",)
    cfg.DATASETS.TEST = ("kitti_val",)
    cfg.INPUT.MASK_FORMAT = "bitmask"  # need this line of code to work with rle masks

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config["model_path"])
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config["num_classes"]

    cfg.SOLVER.IMS_PER_BATCH = config["batch_size"]
    cfg.SOLVER.BASE_LR = config["base_lr"]
    cfg.SOLVER.MAX_ITER = config["max_iter"]

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # VISUALIZE
    dataset_dicts = get_kitti_dataset(config["dataset_path"], "val")
    kitti_metadata = MetadataCatalog.get("kitti_val")

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # load fine-tuned weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # confidence threshold
    predictor = DefaultPredictor(cfg)

    for sample in random.sample(dataset_dicts, 10):
        img = cv2.imread(sample["file_name"])

        # Ground truth
        visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(sample)
        plt.figure(figsize=(15, 7))
        plt.imshow(out.get_image()[:, :, ::-1][..., ::-1])
        plt.title("Ground truth")
        plt.show()

        # Predictions
        predictions = predictor(img)
        visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_metadata, scale=0.5)
        out = visualizer.draw_instance_predictions(predictions["instances"].to("cpu"))
        plt.figure(figsize=(15, 7))
        plt.imshow(out.get_image()[:, :, ::-1][..., ::-1])
        plt.title("Predicted")
        plt.show()

    # EVALUATE VALIDATION
    evaluator = COCOEvaluator("kitti_val", cfg, False, output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "kitti_val")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_task_e.yaml")
    parser.add_argument("--wandb_name", default=None)
    args = parser.parse_args(sys.argv[1:])

    config_path = args.config
    wandb_name = args.wandb_name

    config = load_yaml_config(config_path)

    main(config, wandb_name)


