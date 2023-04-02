import argparse
import sys
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt
from pycocotools.coco import COCO

from week3.utils import load_yaml_config


def visualize_image_and_annotations(img_path: str, annotations: List[Dict], coco: COCO):
    """
    Visualize the image and the annotations
    :param img_path: path to the image
    :param annotations: list of annotations of the image
    :param coco: coco object
    :return: None
    """
    for ann in annotations:
        ann["category"] = coco.loadCats(ann["category_id"])[0]["name"]
    # show the image with the annotations
    fig, ax = plt.subplots(figsize=(10, 8))
    image = Image.open(img_path)
    ax.imshow(image)
    coco.showAnns(annotations, draw_bbox=True)
    for i, ann in enumerate(annotations):
        ax.text(annotations[i]['bbox'][0], annotations[i]['bbox'][1], annotations[i]['category'], style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})
    plt.show()


def get_bbox_and_mask(object_id: int, object_idx: int, coco: COCO, annot: List[Dict]):
    """
    Get the bounding box and the mask of the object
    :param object_id: id of the object
    :param object_idx: index of the object
    :param coco: coco object
    :param annot: annotations of the image
    :return: bounding box and mask of the object
    """

    object_annot = [ann for ann in annot if ann["category_id"] == object_id]
    if not object_annot:
        raise f"No object with id {object_id} found in this image"
    elif len(object_annot) > 1:
        print(f"More than one object with id {object_id} found, extracting the first one found")
    mask = coco.annToMask(object_annot[object_idx])
    bbox = object_annot[object_idx]["bbox"]
    return np.asarray(bbox, dtype=int), mask


def apply_background_and_noise(img_path, bbox, mask):
    img = cv2.imread(img_path)
    out_img = img.copy()
    out_img[:bbox[1], :, :] = 0  # set pixels above the bbox to black
    out_img[bbox[1] + bbox[3]:, :, :] = 0  # set pixels below the bbox to black
    out_img[:, :bbox[0], :] = 0  # set pixels left of the bbox to black
    out_img[:, bbox[0] + bbox[2]:, :] = 0  # set pixels right of the bbox to black

    # apply noise to pixels inside the bbox
    noise = np.random.randint(0, 255, size=(bbox[3], bbox[2], 3))
    out_img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :] = noise
    out_img[mask == 1] = img[mask == 1]  # set pixels inside the mask to the original image

    return out_img


def task_e(cfg):
    img_path = cfg["img_path"]
    annotation_file = cfg["annotations"]
    object_id = cfg["object_id"]
    object_idx = cfg["object_idx"]
    out_black = cfg["out_black"]
    out_roi_black = cfg["out_roi_black"]
    out_black_roi_noise = cfg["out_black_roi_noise"]

    # initialize the COCO API
    coco = COCO(annotation_file)
    # specify the ID of the image
    img_id = int(img_path.split("/")[-1].split(".")[0])
    # get the annotations for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    img_annotations = coco.loadAnns(ann_ids)

    visualize_image_and_annotations(img_path=img_path, annotations=img_annotations, coco=coco)

    bbox, mask = get_bbox_and_mask(object_id=object_id, object_idx=object_idx, coco=coco, annot=img_annotations)

    if out_black['use']:
        out_black_img = cv2.imread(img_path)
        predict_image(out_black_img, effect="default")
        black_img = np.zeros_like(out_black_img)
        black_img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :] = out_black_img[bbox[1]:bbox[1] + bbox[3],
                                                                             bbox[0]:bbox[0] + bbox[2], :]
        plt.imshow(cv2.cvtColor(black_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.tight_layout()
        cv2.imwrite("out_black.png", black_img)
        plt.show()
        predict_image(black_img, effect="out_black")

    if out_black_roi_noise['use']:
        out = apply_background_and_noise(img_path=img_path, bbox=bbox, mask=mask)
        plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.tight_layout()
        cv2.imwrite("out_black_roi_noise.png", out)
        plt.show()
        predict_image(out, effect="out_black_roi_noise")

    if out_roi_black['use']:
        out_roi_black_img = cv2.imread(img_path)
        out_roi_black_img[mask == 0] = 0
        plt.imshow(cv2.cvtColor(out_roi_black_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.tight_layout()
        cv2.imwrite("out_roi_black.png", out_roi_black_img)
        plt.show()
        predict_image(out_roi_black_img, effect="out_roi_black")


def predict_image(result_img: np.ndarray, effect: str):
    models = ["COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
              "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"]
    for model in models:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # confidence threshold
        cfg.MODEL.DEVICE = "cuda"
        predictor = DefaultPredictor(cfg)

        predictions = predictor(result_img)
        visualizer = Visualizer(result_img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
        out = visualizer.draw_instance_predictions(predictions["instances"].to("cpu"))
        out_img_pred = out.get_image()

        plt.imshow(out_img_pred)
        plt.axis("off")
        plt.tight_layout()
        cv2.imwrite(f"{model.split('.')[0].split('/')[-1][:4]}_{effect}.png", out_img_pred[:, :, ::-1])
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config/task_d.yaml")
    args = parser.parse_args(sys.argv[1:])

    config = load_yaml_config(args.config_path)

    task_e(config)
