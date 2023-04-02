"""
This code has been written following the tutorial from Pytorch Metric learning
https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/MetricLossOnly.ipynb
"""

import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from week4.io import load_yaml_config

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import umap
from cycler import cycler

import logging
import argparse
import sys
from torch import optim
from models import HeadlessResnet, Embedder
from torchvision.datasets import ImageFolder


def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
    logging.info(
        "UMAP plot for the {} split and label set {}".format(split_name, keyname)
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
        ax.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=4, label=f"Class {label_set[i]}")
    plt.legend(loc='best', fontsize='large', markerscale=4)
    plt.show()


def get_transforms():
    augmentations = {
        "train":
            transforms.Compose([
                transforms.ColorJitter(brightness=.3, hue=.3),
                transforms.RandomResizedCrop(256, (0.15, 1.0)),
                transforms.RandomRotation(degrees=30),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        "val":
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    }
    return augmentations


def main(cfg):
    logging.getLogger().setLevel(logging.INFO)
    logging.info("VERSION %s" % pytorch_metric_learning.__version__)

    # set datasets
    augmentations = get_transforms()
    train_dataset = ImageFolder(cfg["train_path"], transform=augmentations["train"])
    val_dataset = ImageFolder(cfg["val_path"], transform=augmentations["val"])

    data_labels = [x for _, x in train_dataset.samples]
    class_sampler = samplers.MPerClassSampler(
        labels=data_labels,
        m=cfg["batch_size"] // 8,
        batch_size=cfg["batch_size"],
        length_before_new_iter=len(train_dataset),
    )

    # set model
    trunk_model = HeadlessResnet(cfg["weights_path"]).to(cfg["device"])
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
            "metric_loss": losses.TripletMarginLoss(margin=0.1)
        }
        mining_funcs = {
            "tuple_miner": miners.BatchHardMiner()
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
        visualizer=umap.UMAP(),
        visualizer_hook=visualizer_hook,
        dataloader_num_workers=2,
        accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
    )

    end_of_epoch_hook = hooks.end_of_epoch_hook(
        tester, dataset_dict, model_folder, test_interval=1, patience=1
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
        sampler=class_sampler,
        lr_schedulers={"trunk_scheduler_by_iteration": trunk_scheduler,
                       "embedder_scheduler_by_iteration": embedder_scheduler},
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
    )
    trainer.train(num_epochs=cfg["num_epochs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/task_b.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_path = args.config

    config = load_yaml_config(config_path)

    main(config)
