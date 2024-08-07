"""
Module: PoiModule

This module contains the implementation of the PoiPredictionModule class,
which is a PyTorch Lightning module for predicting points of interest (POI).
It also includes helper functions for creating feature extraction and refinement modules.

Classes:
    - PoiPredictionModule: A PyTorch Lightning module for predicting points of interest.

Functions:
    - create_feature_extraction_module: Creates a feature extraction based on a given configuration.
    - create_refinement_module: Creates a refinement module based on the given configuration.
"""

import numpy as np
import pytorch_lightning as pl
import torch

import modules.FeatureExtractionModules as feat_modules
import modules.RefinementModules as ref_modules


class PoiPredictionModule(pl.LightningModule):
    """A PyTorch Lightning module for POI prediction.

    coarse_config (dict): Configuration for the coarse feature extraction module.
        refinement_config (dict): Configuration for the refinement module.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
        loss_weights (list, optional): Weights for the feature extraction and refinement losses. Defaults to None.
        optimizer (str, optional): Optimizer algorithm. Defaults to "AdamW".
        scheduler_config (dict, optional): Configuration for the learning rate scheduler. Defaults to None.
        feature_freeze_patience (int, optional): Number of epochs without improvement before freezing the feature extraction module. Defaults to None.
    Attributes:
        feature_extraction_module (Module): The feature extraction module.
        refinement_module (Module): The refinement module.
        lr (float): Learning rate for the optimizer.
        loss_weights (Tensor): Weights for the feature extraction and refinement losses.
        feature_freeze_patience (int): Number of epochs without improvement before freezing the feature extraction module.
        best_feature_loss (float): Best feature loss achieved during validation.
        val_feature_loss_outputs (list): List of feature loss values during validation.
        epochs_without_improvement (int): Number of epochs without improvement during validation.
        feature_extactor_frozen (bool): Flag indicating if the feature extraction module is frozen.
        optimizer (str): Optimizer algorithm.
        scheduler_config (dict): Configuration for the learning rate scheduler.
    Methods:
        forward(*args, **kwargs): Forward pass of the module.
        training_step(*args, **kwargs): Training step of the module.
        validation_step(*args, **kwargs): Validation step of the module.
        on_validation_epoch_end(): Callback function called at the end of each validation epoch.
        configure_optimizers(): Configures the optimizer and learning rate scheduler.
        calculate_metrics(batch, mode): Calculates metrics for the given batch and mode.
        freeze_feature_extractor(): Freezes the feature extraction module.
    """

    def __init__(
        self,
        coarse_config,
        refinement_config,
        lr=1e-4,
        loss_weights=None,
        optimizer="AdamW",
        scheduler_config=None,
        feature_freeze_patience=None,
    ):
        super().__init__()
        if loss_weights is None:
            loss_weights = [1, 1]
        self.feature_extraction_module = create_feature_extraction_module(coarse_config)
        self.refinement_module = create_refinement_module(refinement_config)
        self.lr = lr
        self.loss_weights = torch.tensor(loss_weights) / torch.sum(
            torch.tensor(loss_weights)
        )
        self.feature_freeze_patience = feature_freeze_patience
        self.best_feature_loss = np.inf
        self.val_feature_loss_outputs = []
        self.epochs_without_improvement = 0
        self.feature_extactor_frozen = False
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        """Performs the forward pass of the module.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The processed batch after passing through the feature extraction and refinement modules.

        Raises:
            ValueError: If batch input is not provided.
        """

        batch = args[0] if args else kwargs.get("batch")
        if batch is None:
            raise ValueError("Batch input is required for the forward pass.")

        batch = self.feature_extraction_module(batch)
        batch = self.refinement_module(batch)
        return batch

    def training_step(self, *args, **kwargs):
        batch = args[0] if args else kwargs.get("batch")
        if batch is None:
            raise ValueError("Batch input is required for the forward pass.")
        batch = self(batch)

        # Calculate the feature extraction loss
        feature_loss = self.feature_extraction_module.calculate_loss(batch)
        # Calculate the refinement loss
        refinement_loss = self.refinement_module.calculate_loss(batch)
        loss = (
            feature_loss * self.loss_weights[0] + refinement_loss * self.loss_weights[1]
        )

        metrics = self.calculate_metrics(batch, "train")
        batch_size = batch["input"].shape[0]

        self.log("train_loss", loss, on_epoch=True, batch_size=batch_size)
        self.log_dict(metrics, on_epoch=True, on_step=False, batch_size=batch_size)

        return loss

    def validation_step(self, *args, **kwargs):
        batch = args[0] if args else kwargs.get("batch")
        if batch is None:
            raise ValueError("Batch input is required for the forward pass.")
        # Calculate the feature extraction loss
        feature_loss = self.feature_extraction_module.calculate_loss(batch)
        # Calculate the refinement loss
        refinement_loss = self.refinement_module.calculate_loss(batch)
        loss = (
            feature_loss * self.loss_weights[0] + refinement_loss * self.loss_weights[1]
        )

        metrics = self.calculate_metrics(batch, "val")
        batch_size = batch["input"].shape[0]

        self.val_feature_loss_outputs.append(feature_loss)

        self.log("val_feature_loss", feature_loss, on_epoch=True, batch_size=batch_size)
        self.log(
            "val_refinement_loss", refinement_loss, on_epoch=True, batch_size=batch_size
        )
        self.log("val_loss", loss, on_epoch=True, batch_size=batch_size)
        self.log_dict(metrics, on_epoch=True, on_step=False, batch_size=batch_size)

        return loss

    def on_validation_epoch_end(self):
        # Check if the feature extraction module should be frozen
        if self.feature_extactor_frozen:
            return

        avg_feature_loss = torch.stack(self.val_feature_loss_outputs).mean()
        self.val_feature_loss_outputs.clear()

        if self.feature_freeze_patience is not None:
            if avg_feature_loss < self.best_feature_loss:
                self.best_feature_loss = avg_feature_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if (
                    self.epochs_without_improvement >= self.feature_freeze_patience
                    and not self.feature_extactor_frozen
                ):
                    self.freeze_feature_extractor()
                    self.feature_extactor_frozen = True
                    print("Feature extraction module frozen")

    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.optimizer)
        optimizer = optimizer_class(self.parameters(), lr=self.lr)

        if self.scheduler_config:
            scheduler_class = getattr(
                torch.optim.lr_scheduler, self.scheduler_config["type"]
            )
            scheduler = scheduler_class(optimizer, **self.scheduler_config["params"])

            scheduler_config = {"scheduler": scheduler, "interval": "epoch"}
            if "monitor" in self.scheduler_config:
                scheduler_config["monitor"] = self.scheduler_config["monitor"]

            return [optimizer], [scheduler_config]

        return optimizer

    def calculate_metrics(self, batch, mode):
        """Calculates metrics for the given batch and mode.

        Parameters:
            batch (Tensor): The input batch.
            mode (str): The mode of calculation.

        Returns:
            dict: A dictionary containing the calculated metrics.
        """

        feature_metrics = self.feature_extraction_module.calculate_metrics(batch, mode)
        refinement_metrics = self.refinement_module.calculate_metrics(batch, mode)

        return {**feature_metrics, **refinement_metrics}

    def freeze_feature_extractor(self):
        """Freezes the feature extraction module by setting the `requires_grad`
        attribute of all its parameters to False.

        This prevents the feature extraction module from being updated during training.

        Args:
            None

        Returns:
            None
        """
        self.log("feature_frozen", True, on_epoch=True)
        for param in self.feature_extraction_module.parameters():
            param.requires_grad = False


def create_feature_extraction_module(config):
    """Create a feature extraction module based on the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters for the module.

    Returns:
        module_type: An instance of the feature extraction module.

    Raises:
        ValueError: If the provided module type is unknown.
    """

    module_type = getattr(feat_modules, config["type"])
    if module_type is None:
        raise ValueError(f"Unknown feature extraction module type: {config['type']}")

    return module_type(**config["params"])


def create_refinement_module(config):
    """Create a refinement module based on the given configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters for the module.

    Returns:
        object: An instance of the refinement module.

    Raises:
        ValueError: If the specified module type is unknown.

    Example:
        config = {
            "type": "SomeModule",
            "params": {
                "param1": value1,
                "param2": value2
            }
        }
        module = create_refinement_module(config)
    """

    module_type = getattr(ref_modules, config["type"])
    if module_type is None:
        raise ValueError(f"Unknown refinement module type: {config['type']}")

    return module_type(**config["params"])
