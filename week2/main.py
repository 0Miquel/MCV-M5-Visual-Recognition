from detector import Detector


detector = Detector("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
detector.inference("./input.jpg")
