from detectron2.config import get_cfg
from detectron2 import model_zoo


def custom_config():
    cfg = get_cfg()

    # get configuration from model_zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.DEVICE = "cuda"
    # cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    # cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    # cfg.MODEL.RESNETS.DEPTH = 34

    # Solver
    cfg.SOLVER.BASE_LR = 0.0002
    cfg.SOLVER.MAX_ITER = 40000
    cfg.SOLVER.STEPS = (20, 10000, 20000)
    cfg.SOLVER.gamma = 0.5
    cfg.SOLVER.IMS_PER_BATCH = 4

    # Test
    cfg.TEST.DETECTIONS_PER_IMAGE = 20

    # INPUT
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)

    # DATASETS
    cfg.DATASETS.TEST = ()
    cfg.DATASETS.TRAIN = ('kitti_train',)

    # DATASETS
    cfg.OUTPUT_DIR = "outputs"

    return cfg



