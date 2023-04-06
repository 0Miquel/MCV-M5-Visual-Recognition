"""
This code has been written following the tutorial from Pytorch Metric learning
https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/MetricLossOnly.ipynb
"""

import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import distances, reducers
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import torch

from week4.i_o import load_yaml_config

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import umap
from cycler import cycler

import logging
import argparse
import sys
from torch import optim
from models import HeadlessResnet, Embedder, Fusion
from torchvision.datasets import ImageFolder

from torchvision.datasets import CocoDetection
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch.nn as nn
import torch.optim as optim

import wandb

#pip install record-keeper tensorboard
#pip install faiss

def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
    idx_to_class = {0: 'Opencountry', 1: 'coast', 2: 'forest', 3: 'highway', 4: 'inside_city',
                    5: 'mountain', 6: 'street', 7: 'tallbuilding'}

    logging.info(
        f"UMAP plot for the {split_name} split and epoch {args[0]}"
    )
    label_set = np.unique(labels)
    num_classes = len(label_set)
    fig, ax = plt.subplots(figsize=(20, 15))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        idx = labels == label_set[i]
        ax.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=10,
                label=f"{idx_to_class[label_set[i]]}")
    plt.legend(loc='best', fontsize='large', markerscale=1)
    plt.title(f"UMAP plot for the {split_name} split and epoch {args[0]}")
    plt.savefig('umap.jpg')
    plt.show()


def get_transforms():
    augmentations = {
        "train":
            transforms.Compose([
                # transforms.ColorJitter(brightness=.3, hue=.3),
                # transforms.RandomResizedCrop(256, (0.15, 1.0)),
                # transforms.RandomRotation(degrees=30),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        "val":
            transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    }
    return augmentations


def main(cfg):
    # wandb.tensorboard.patch(root_logdir="example_tensorboard")
    # wandb.init(project='test', sync_tensorboard=True)

    logging.getLogger().setLevel(logging.INFO)
    logging.info("VERSION %s" % pytorch_metric_learning.__version__)

    # set datasets
    augmentations = get_transforms()
    train_dataset = CocoDetection(root=cfg["train_path"], annFile=cfg["train_ann_file"], transform=augmentations)
    val_dataset = CocoDetection(root=cfg["val_path"], annFile=cfg["val_ann_file"], transform=augmentations)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True,num_workers=4)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    # train_dataset = ImageFolder(cfg["train_path"], transform=augmentations["train"])
    # val_dataset = ImageFolder(cfg["val_path"], transform=augmentations["val"])

    # data_labels = [x for _, x in train_dataset.samples]
    # class_sampler = samplers.MPerClassSampler(
    #     labels=data_labels,
    #     m=cfg["batch_size"] // 8,
    #     batch_size=cfg["batch_size"],
    #     length_before_new_iter=len(train_dataset),
    # )

    # set model
    trunk_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(cfg["device"])
    num_classes = 91  # COCO dataset has 80 classes + 1 background class
    in_features = trunk_model.roi_heads.box_predictor.cls_score.in_features
    trunk_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # trunk_model = HeadlessResnet(cfg["weights_path"]).to(cfg["device"])
    trunk_optimizer = optim.Adam(trunk_model.parameters(), cfg["lr"])
    trunk_scheduler = optim.lr_scheduler.OneCycleLR(trunk_optimizer, max_lr=cfg["max_lr"],
                                                    steps_per_epoch=len(train_dataset) // cfg["batch_size"],
                                                    epochs=cfg["num_epochs"])

    embedder_model = Embedder(512, cfg["embedder_size"]).to(cfg["device"])
    embedder_optimizer = optim.Adam(embedder_model.parameters(), cfg["lr"])
    embedder_scheduler = optim.lr_scheduler.OneCycleLR(embedder_optimizer, max_lr=cfg["max_lr"],
                                                       steps_per_epoch=len(train_dataset) // cfg["batch_size"],
                                                       epochs=cfg["num_epochs"])

    # set loss function
    if cfg["loss"] == "contrastive":
        loss_funcs = {
            "metric_loss": losses.ContrastiveLoss()
        }
        mining_funcs = {
            "tuple_miner": miners.PairMarginMiner()
        }

    else:  # triplet loss
        loss_funcs = {
            "metric_loss": losses.TripletMarginLoss(margin=0.2)
        }
        mining_funcs = {
            "tuple_miner": miners.TripletMarginMiner(margin=0.2, type_of_triplets="semihard")
        }

    record_keeper, _, _ = logging_presets.get_record_keeper(
        "example_logs", "example_tensorboard"
    )
    hooks = logging_presets.get_hook_container(record_keeper)
    dataset_dict = {"val": val_dataset}
    model_folder = "example_saved_models"

    # Create the tester
    tester = testers.GlobalEmbeddingSpaceTester(
        end_of_testing_hook=hooks.end_of_testing_hook,
        # visualizer=umap.UMAP(),
        # visualizer_hook=visualizer_hook,
        dataloader_num_workers=2,
        accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
    )

    end_of_epoch_hook = hooks.end_of_epoch_hook(
        tester, dataset_dict, model_folder, test_interval=cfg["test_interval"], patience=cfg["patience"]
    )

    # Create the trainer
    trainer = trainers.MetricLossOnly(
        models={"trunk": trunk_model,
                "embedder": embedder_model},
        optimizers={"trunk_optimizer": trunk_optimizer,
                    "embedder_optimizer": embedder_optimizer},
        batch_size=cfg["batch_size"],
        loss_funcs=loss_funcs,
        mining_funcs=mining_funcs,
        dataset=train_dataset,
        data_device=cfg["device"],
        # sampler=class_sampler,
        lr_schedulers={"trunk_scheduler_by_iteration": trunk_scheduler,
                       "embedder_scheduler_by_iteration": embedder_scheduler},
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
    )
    trainer.train(num_epochs=cfg["num_epochs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/task_e.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_path = args.config

    config = load_yaml_config(config_path)

    main(config)
