import argparse
import sys

from src.models import EmbeddingNetImage, EmbeddingNetText
from src.utils_io import load_yaml_config
from src.datasets import TripletIm2Text, TripletText2Im


def main(cfg):
    dataset = TripletIm2Text(cfg['train_captions'], cfg['train_dir'])
    a = dataset[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/task_a.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_path = args.config

    config = load_yaml_config(config_path)

    main(config)
