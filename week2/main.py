from dataset import get_kitti_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import cv2
import random

##################################################################################
# Train data visualization
##################################################################################

for d in ["train", "val"]:
    DatasetCatalog.register("kitti_" + d, lambda d=d: get_kitti_dataset("../dataset/KITTI-MOTS/", d))
    MetadataCatalog.get("kitti_" + d).set(thing_classes=['Car', 'Pedestrian'])
kitti_metadata = MetadataCatalog.get("kitti_train")

dataset_dicts = get_kitti_dataset("../dataset/KITTI-MOTS/", "train")
for d in random.sample(dataset_dicts, 5):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    plt.figure(figsize=(15,7))
    plt.imshow(out.get_image()[:, :, ::-1][..., ::-1])
    plt.show()

##################################################################################
