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
    for instance, sequence in zip(instances, sequences):
        # iterate over sequences
        annotations = load_txt(instance)
        images = glob.glob(sequence+"/*.png")

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
