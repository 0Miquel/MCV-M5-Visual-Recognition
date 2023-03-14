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


def inference():
    for d in ["train", "val"]:
        DatasetCatalog.register("kitti_" + d, lambda d=d: get_kitti_dataset("../dataset/KITTI-MOTS/", d))
        MetadataCatalog.get("kitti_" + d).set(thing_classes=['Car', 'Pedestrian'])
    kitti_metadata = MetadataCatalog.get("kitti_train")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    # cfg.INPUT.MASK_FORMAT = "bitmask"

    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_kitti_dataset("../dataset/KITTI-MOTS/", "train")
    for d in random.sample(dataset_dicts, 10):
        img = cv2.imread(d["file_name"])
        predictions = predictor(img)
        visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_metadata, scale=0.5)
        out = visualizer.draw_instance_predictions(predictions["instances"].to("cpu"))
        plt.figure(figsize=(15, 7))
        plt.imshow(out.get_image()[:, :, ::-1][..., ::-1])
        plt.show()


if __name__ == "__main__":
    inference()
