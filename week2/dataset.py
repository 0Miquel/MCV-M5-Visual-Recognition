from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import os
import glob
from io2 import *


def get_kitti_dataset(path):
    instances = path+"instances_txt"
    sequences = path+"training/image_02"

    instances = glob.glob(instances+"/*.txt")
    sequences = glob.glob(sequences+"/*")

    for instance, sequence in zip(instances, sequences):
        annotations = open(instance, "r").readlines()
        images = glob.glob(sequence+"/*.png")

        for annotation in annotations:
            split_annotation = annotation.split(" ")
            image_id = int(split_annotation[0])
            image = images[image_id]


# for d in ["train", "val"]:
#     DatasetCatalog.register("kitti_" + d, get_kitti_dataset("../dataset/KITTI-MOTS/"))
#     MetadataCatalog.get("kitti_" + d).set(thing_classes=['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
#                                                          'Cyclist', 'Tram', 'Misc', 'DontCare'])
# kitti_metadata = MetadataCatalog.get("kitti_train")


annotations = load_sequences("../dataset/KITTI-MOTS/instances_txt", [str(a).zfill(4) for a in range(21)])
images = load_sequences("../dataset/KITTI-MOTS/training/image_02", [str(a).zfill(4) for a in range(21)])
print("Hi")