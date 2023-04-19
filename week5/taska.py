from week5.i_o import load_yaml_config
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import umap
from cycler import cycler
import torch
import logging
import argparse
import sys
from torch import optim
from torchvision.datasets import ImageFolder
from tqdm import tqdm

def main(cfg):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/taska.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_path = args.config

    config = load_yaml_config(config_path)

    main(config)
