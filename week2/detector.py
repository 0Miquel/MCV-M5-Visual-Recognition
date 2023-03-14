"""
Code to test if detectron2 works
"""


from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import cv2
import numpy as np


class Detector:
    def __init__(self, model_cfg):
        self.cfg = get_cfg()

        self.cfg.merge_from_file(model_zoo.get_config_file(model_cfg))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_cfg)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda"

        self.predictor = DefaultPredictor(self.cfg)

    def inference(self, im_path):
        im = cv2.imread(im_path)[:, :, ::-1]
        v = Visualizer(im, MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)

        outputs = self.predictor(im)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        cv2.imshow("Result", out.get_image()[..., ::-1])
        cv2.waitKey(0)


detector = Detector("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
detector.inference("./input.jpg")
