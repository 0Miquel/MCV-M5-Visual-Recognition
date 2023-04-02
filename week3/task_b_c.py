from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import cv2


###############################################################
# PARAMETERS YOU CAN CHANGE
###############################################################
# source image where you extract the object
object_img_path = "../dataset/coco2017/val2017/000000105912.jpg"
# object id we want to extract and paste
object_id = 11  # check COCO objects id in https://github.com/pjreddie/darknet/blob/master/data/coco.names
# number of object to extract in case there are more than one of the same category in the image
object_idx = 0
# destination image where you paste the object
source_img_path = "../result_img.png"
# 000000086755.jpg # people in snow
# 000000000139.jpg # living room
# 000000020247.jpg # bears
# 000000039551.jpg # tennis player
# 000000001761.jpg # planes
# 000000007816.jpg # motorcycle
# 000000050679.jpg # huge orange
# 000000089761.jpg # strange thing
# 000000105912.jpg # boca de incendios
# specify the path to the annotation file
annFile = '../dataset/coco2017/annotations/instances_val2017.json'
# position of pasted object in destination image
position = (200, 0)


###############################################################
# GET OBJECT MASK USING COCO API
###############################################################
# initialize the COCO API
coco = COCO(annFile)
# specify the ID of the image
img_id = int(object_img_path.split("/")[-1].split(".")[0])
# get the information for the image
img_info = coco.loadImgs(img_id)[0]
# get the annotations for the image
ann_ids = coco.getAnnIds(imgIds=img_id)
anns = coco.loadAnns(ann_ids)
# get the segmentation mask for the object in the image
object_anns = [ann for ann in anns if ann["category_id"] == object_id]
if not object_anns:
    raise f"No object with id {object_id} found in this image"
elif len(object_anns) > 1:
    print(f"More than one object with id {object_id} found, extracting the first one found")
mask = coco.annToMask(object_anns[object_idx])


###############################################################
# CREATE RESULTING IMAGE
###############################################################
# get the object image
object_img = Image.open(object_img_path)
plt.imshow(np.array(object_img))
plt.axis("off")
plt.tight_layout()
plt.show()
# get the destination image
source_img = Image.open(source_img_path)
plt.imshow(np.array(source_img))
plt.axis("off")
plt.tight_layout()
plt.show()
# transform to mask to PIL
mask = Image.fromarray(mask*255)
# Apply the mask to the object image
object_masked_img = object_img.convert("RGBA")
object_masked_img.putalpha(mask)
source_img.paste(object_masked_img, position, object_masked_img)
# # save image
source_img.save("new.png", format="png")
# plot resulting image
result_img = np.array(source_img)
plt.imshow(result_img)
plt.axis("off")
plt.tight_layout()
plt.show()

cv2.imwrite(f"result_img.png", result_img[:, :, ::-1])

###############################################################
# INFERE RESULTING IMAGE WITH FASTER AND MASK MODELS
###############################################################
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
    outimgpred = out.get_image()[:, :, ::-1]

    plt.imshow(outimgpred)
    plt.axis("off")
    plt.tight_layout()
    cv2.imwrite(f"{model.split('.')[0].split('/')[-1]}.png", outimgpred[:, :, ::-1])
    plt.show()
