from typing import Union, Dict

import wandb
from tqdm import tqdm
from source.datasets import get_dataloaders
from source.losses import get_loss
from source.models import get_model
from source.optimizers import get_optimizer
from source.schedulers import get_scheduler
from source.utils import *
import math
import time
from omegaconf import OmegaConf
from torchsummary import summary


def train(wandb_name=None, cfg=None):
    trainer = Trainer(cfg, wandb_name)
    trainer.fit()


class Trainer:
    def __init__(self, config, wandb_name):
        self.log = False
        if wandb_name is not None:
            # if wandb project name is set or if config is none which means that we are executing a sweep
            self.log = True
            self.logger = get_logger(config, wandb_name)
            config = self.logger.cfg
        else:
            config = OmegaConf.to_object(config)

        trainer_config = config["trainer"]
        self.metrics = trainer_config["metrics"]
        self.n_epochs = trainer_config["n_epochs"]
        self.device = trainer_config["device"]
        self.model_path = trainer_config["model_path"]
        self.task = trainer_config["task"]
        self.grad_cam = trainer_config["grad_cam"]

        dataloaders = get_dataloaders(config['dataset'])
        self.train_dl = dataloaders["train"]
        self.val_dl = dataloaders["val"]
        self.loss = get_loss(config['loss'])
        model = get_model(config['model'])
        self.model = model.to(self.device)
        summary(model, input_size=(3, 224, 224), device=self.device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.log({"num_params": num_params})
        self.optimizer = get_optimizer(config['optimizer'], self.model)
        self.scheduler = get_scheduler(config['scheduler'], self.optimizer, len(self.train_dl),
                                       n_epochs=self.n_epochs) if "scheduler" in config.keys() else None

    def train_epoch(self, epoch):
        self.model.train()
        init_exec_params(self.metrics)
        # use tqdm to track progress
        with tqdm(self.train_dl, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} train")
            # Iterate over data.
            for inputs, targets, og_imgs in tepoch:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward
                outputs = self.model(inputs)
                # loss
                loss = self.loss(outputs, targets)
                # backward
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                # compute metrics for this epoch +  current lr and loss
                metrics = compute_metrics(self.metrics, outputs, targets, inputs, loss, self.optimizer)
                tepoch.set_postfix(**metrics)
        return metrics["loss"]

    def val_epoch(self, epoch):
        self.model.eval()
        init_exec_params(self.metrics)
        with torch.no_grad():
            # use tqdm to track progress
            with tqdm(self.val_dl, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{self.n_epochs} val")
                # Iterate over data.
                for inputs, targets, og_imgs in tepoch:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    # predict
                    outputs = self.model(inputs)
                    # loss
                    loss = self.loss(outputs, targets)
                    # compute metrics for this epoch +  current lr and loss
                    metrics = compute_metrics(self.metrics, outputs, targets, inputs, loss)
                    tepoch.set_postfix(**metrics)
        if self.log:
            # predict
            outputs = self.model(inputs)
            # loss
            self.loss(outputs, targets)
            grad_cams = self.get_grad_cam(inputs) if self.grad_cam else None
            self.logger.add(og_imgs, outputs, targets, metrics, "val", grad_cam=grad_cams)
        return metrics["loss"]

    def fit(self):
        since = time.time()
        best_loss = math.inf
        for epoch in range(self.n_epochs):
            self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
            if self.log:
                self.logger.upload()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        if self.log:
            self.logger.log_model(self.model_path)
            self.logger.finish()

    def get_grad_cam(self, x: torch.Tensor) -> np.ndarray:
        """
        Computes the grad cam for the given input. Assumes a batch of data.
        Extracted from: https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
        :param x: input tensor (batch_size, channels, height, width)
        :return: grad cam (batch_size, height, width)
        """
        assert self.grad_cam, "Grad cam is not enabled"
        # pull the gradients out of the model
        gradients = self.model.get_activations_gradient()
        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        # get the activations of the last convolutional layer
        activations = self.model.get_activations(x).detach()
        # weight the channels by corresponding gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap.cpu(), 0)
        # normalize the heatmap
        heatmap /= torch.max(heatmap)
        return heatmap

