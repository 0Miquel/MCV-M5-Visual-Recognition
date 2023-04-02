import os
import random
from datetime import datetime

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from dataset import create_detectron_dataset


def main():
    now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + "_task_a"
    os.makedirs('../results', exist_ok=True)
    os.makedirs(f'../results/{now}', exist_ok=True)
    paths = [f'../results/{now}/faster', f'../results/{now}/mask']
    models = ["COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
              "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"]
    dataset_path = "../dataset/out_of_context/"

    for path, model in zip(paths, models):
        # Create Predictor
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # confidence threshold
        cfg.MODEL.DEVICE = "cuda"
        predictor = DefaultPredictor(cfg)

        dataset_dicts = create_detectron_dataset(dataset_path)
        random.seed(42)
        for i, sample in enumerate(random.sample(dataset_dicts, 10)):
            img = cv2.imread(sample["file_name"])
            # Predictions
            predictions = predictor(img)
            visualizer = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
            out = visualizer.draw_instance_predictions(predictions["instances"].to("cpu"))
            outimgpred = out.get_image()[:, :, ::-1]

            os.makedirs(path, exist_ok=True)
            cv2.imwrite(f"{path}/pred{str(i).zfill(3)}.jpg", outimgpred)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", default="config/inference_and_eval.yaml")
    # args = parser.parse_args(sys.argv[1:])
    #
    # config_path = args.config
    #
    # config = load_yaml_config(config_path)
    main()
