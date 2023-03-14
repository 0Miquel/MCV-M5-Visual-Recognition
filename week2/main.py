from dataset import get_kitti_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import cv2
import random
from detectron2.engine import DefaultPredictor
from config import custom_config


for d in ["train"]:
    DatasetCatalog.register("kitti_" + d, lambda d=d: get_kitti_dataset("../dataset/KITTI-MOTS/", d))
    MetadataCatalog.get("kitti_" + d).set(thing_classes=['Car', 'Pedestrian'])

kitti_metadata = MetadataCatalog.get("kitti_train")

cfg = custom_config()

predictor = DefaultPredictor(cfg)

dataset_dicts = get_kitti_dataset("../dataset/KITTI-MOTS/", "train")

for d in random.sample(dataset_dicts, 5):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    visualizer = Visualizer(im[:, :, ::-1], metadata=kitti_metadata, scale=0.5)

    out_pred = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(15, 7))
    plt.imshow(out_pred.get_image()[:, :, ::-1][..., ::-1])
    plt.title("Prediction")
    plt.show()

    out_gt = visualizer.draw_dataset_dict(d)
    plt.figure(figsize=(15, 7))
    plt.imshow(out_gt.get_image()[:, :, ::-1][..., ::-1])
    plt.title("Ground truth")
    plt.show()
