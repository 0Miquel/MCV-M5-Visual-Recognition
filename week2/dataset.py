from detectron2.structures import BoxMode
from io_kitti import *
import pycocotools.mask as maskUtils
import cv2


def get_kitti_dataset(path, phase):
    instances = path+"instances_txt"
    sequences = path+"training/image_02"

    instances = glob.glob(instances+"/*.txt")
    instances = sorted(instances, key=lambda x: x[-8:])
    sequences = glob.glob(sequences+"/*")
    sequences = sorted(sequences, key=lambda x: x[-5:])

    dataset_dicts = []
    image_id = 0
    # Sequences 2, 6, 7, 8, 10, 13, 14, 16 and 18 were chosen for the validation set,
    # the remaining sequences for the training set.
    # Following the idea of the paper and slides https://arxiv.org/pdf/1902.03604.pdf
    train_sequences = [sequence for i, sequence in enumerate(sequences) if i not in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
    val_sequences = [sequences[i] for i in [2, 6, 7, 8, 10, 13, 14, 16, 18]]

    train_instances = [instance for i, instance in enumerate(instances) if i not in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
    val_instances = [instances[i] for i in [2, 6, 7, 8, 10, 13, 14, 16, 18]]

    sequences = train_sequences if phase == "train" else val_sequences
    instances = train_instances if phase == "train" else val_instances

    for instance, sequence in zip(instances, sequences):
        # iterate over sequences
        annotations = load_txt(instance)
        images = glob.glob(sequence+"/*.png")
        images = sorted(images, key=lambda x: x[-8:])

        for idx, annotation in annotations.items():
            # iterate over images in sequence
            record = {}

            image_filename = images[idx]
            height, width = cv2.imread(image_filename).shape[:2]

            record["file_name"] = image_filename
            record["image_id"] = image_id
            image_id += 1
            record["height"] = height
            record["width"] = width

            objs = []
            for v in annotation:
                if v.class_id != 10:
                    bbox = maskUtils.toBbox(v.mask).tolist()
                    # iterate over annotations in image
                    obj = {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": v.mask,
                        "category_id": v.class_id-1
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts


def get_kitti_dataset_COCO_id(path, phase):
    COCO_classes = {
        0: 50,  # Background anywhere
        1: 2,  # Car to Car
        2: 0,  # Pedestrian to Person
    }
    instances = path+"instances_txt"
    sequences = path+"training/image_02"

    instances = glob.glob(instances+"/*.txt")
    instances = sorted(instances, key=lambda x: x[-8:])
    sequences = glob.glob(sequences+"/*")
    sequences = sorted(sequences, key=lambda x: x[-5:])

    dataset_dicts = []
    image_id = 0
    # Sequences 2, 6, 7, 8, 10, 13, 14, 16 and 18 were chosen for the validation set,
    # the remaining sequences for the training set.
    # Following the idea of the paper and slides https://arxiv.org/pdf/1902.03604.pdf
    train_sequences = [sequence for i, sequence in enumerate(sequences) if i not in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
    val_sequences = [sequences[i] for i in [2, 6, 7, 8, 10, 13, 14, 16, 18]]

    train_instances = [instance for i, instance in enumerate(instances) if i not in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
    val_instances = [instances[i] for i in [2, 6, 7, 8, 10, 13, 14, 16, 18]]

    sequences = train_sequences if phase == "train" else val_sequences
    instances = train_instances if phase == "train" else val_instances

    for instance, sequence in zip(instances, sequences):
        # iterate over sequences
        annotations = load_txt(instance)
        images = glob.glob(sequence+"/*.png")
        images = sorted(images, key=lambda x: x[-8:])

        for idx, annotation in annotations.items():
            # iterate over images in sequence
            record = {}

            image_filename = images[idx]
            height, width = cv2.imread(image_filename).shape[:2]

            record["file_name"] = image_filename
            record["image_id"] = image_id
            image_id += 1
            record["height"] = height
            record["width"] = width

            objs = []
            for v in annotation:
                if v.class_id != 10:
                    bbox = maskUtils.toBbox(v.mask).tolist()
                    # iterate over annotations in image
                    obj = {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": v.mask,
                        "category_id": COCO_classes[v.class_id]
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts
