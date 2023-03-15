from dataset import get_kitti_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import cv2
import random
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import argparse
import sys
from utils import *

def main(config, wandb_name):
    # Register and get KITTI dataset
    for phase in ["train", "val"]:
        DatasetCatalog.register("kitti_" + phase, lambda phase=phase: get_kitti_dataset(config["dataset_path"], phase))
        MetadataCatalog.get("kitti_" + phase).set(thing_classes=['Car', 'Pedestrian'])
    kitti_metadata = MetadataCatalog.get("kitti_train")
    dataset_dicts = get_kitti_dataset(config["dataset_path"], "train")

    # Create Predictor
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config["model_path"]))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config["model_path"])
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # confidence threshold
    cfg.MODEL.DEVICE = "cuda"
    predictor = DefaultPredictor(cfg)

    # get COCO labels that are also in KITTI
    coco_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    coco_labels = coco_metadata.thing_classes
    retain = [coco_labels.index("car"), coco_labels.index("person")]

    # visualize KITTI-MOTS ground truth and predictions
    for sample in random.sample(dataset_dicts, 10):
        img = cv2.imread(sample["file_name"])

        #Ground truth
        visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(sample)
        plt.figure(figsize=(15, 7))
        plt.imshow(out.get_image()[:, :, ::-1][..., ::-1])
        plt.title("Ground truth")
        plt.show()

        #Predictions
        predictions = predictor(img)
        visualizer = Visualizer(img[:, :, ::-1], metadata=coco_metadata, scale=0.5)
        instances = predictions["instances"]
        kitti_instances = instances[(instances.pred_classes == retain[0]) | (instances.pred_classes == retain[1])]
        out = visualizer.draw_instance_predictions(kitti_instances.to("cpu"))
        plt.figure(figsize=(15, 7))
        plt.imshow(out.get_image()[:, :, ::-1][..., ::-1])
        plt.title("Predicted")
        plt.show()

    #Evaluation
    evaluator = COCOEvaluator("kitti_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "kitti_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_taskc.yaml")
    parser.add_argument("--wandb_name", default=None)
    args = parser.parse_args(sys.argv[1:])

    config_path = args.config
    wandb_name = args.wandb_name

    config = load_yaml_config(config_path)

    main(config, wandb_name)
