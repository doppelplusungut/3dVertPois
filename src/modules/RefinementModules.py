import pytorch_lightning as pl
import torch
import torch.nn as nn
from monai.networks.nets.regressor import Regressor

from loss.loss_modules import get_loss_fn
from models.PatchExtraction import PatchExtractor
from models.PoiTransformer import PoiTransformer
from utils.misc import surface_project_coords


class RefinementModule(pl.LightningModule):
    """Defines a generic refinement module.

    The module is expected to have a forward method that takes a batch of data in
    dictionary format, containing the extracted features and the coarse predictions, and
    returns the batch. It needs to implement a calculate_loss method.
    """

    def __init__(self):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, batch):
        raise NotImplementedError

    def training_step(self, batch):
        batch = self(batch)
        loss = self.calculate_loss(batch)
        metrics = self.calculate_metrics(batch, "train")
        self.log_dict(metrics, on_epch=True)

        return loss

    def validation_step(self, batch):
        batch = self(batch)
        loss = self.calculate_loss(batch)
        metrics = self.calculate_metrics(batch, "val")
        self.log_dict(metrics, on_epch=True)

        return loss

    def calculate_loss(self, batch):
        raise NotImplementedError

    def calculate_metrics(self, batch, mode):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class Identity(RefinementModule):
    """Defines a refinement module that does nothing.

    This is useful for testing the feature extraction module in isolation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, batch):
        return batch

    def calculate_loss(self, batch):
        return 0

    def calculate_metrics(self, batch, mode):
        return {}


class FeatureTransformer(nn.Module):
    """This module refines the predictions of a given coarse model."""

    def __init__(
        self,
        n_landmarks: int,
        poi_feature_l: int,
        coord_embedding_l: int,
        poi_embedding_l: int,
        vert_embedding_l: int,
        loss_fn: str,
        mlp_dim: int = 1024,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        lr: float = 1e-5,
    ):
        super().__init__()

        self.refinement_model = PoiTransformer(
            poi_feature_l=poi_feature_l,
            coord_embedding_l=coord_embedding_l,
            poi_embedding_l=poi_embedding_l,
            vert_embedding_l=vert_embedding_l,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            n_landmarks=n_landmarks,
            dropout_rate=dropout,
        )

        self.loss_fn = get_loss_fn(loss_fn)

        self.lr = lr

    def forward(self, batch):
        coarse_preds = batch["coarse_preds"]
        poi_indices = batch["target_indices"]
        vertebra = batch["vertebra"]
        poi_features = batch["coarse_features"]
        offsets = self.refinement_model(
            batch["coarse_preds"], poi_indices, vertebra, poi_features
        )
        batch["offsets"] = offsets
        batch["refined_preds"] = coarse_preds.detach() + offsets

        return batch

    def calculate_loss(self, batch):
        return self.loss_fn(batch["refined_preds"], batch["target"], batch["loss_mask"])

    def calculate_metrics(self, batch, mode):
        metrics = {}
        loss_mask = batch["loss_mask"]  # (batch_size, n_landmarks)
        fine_preds = batch["refined_preds"]  # (batch_size, n_landmarks, 3)
        target = batch["target"]  # (batch_size, n_landmarks, 3)
        target_indices = batch["target_indices"]  # (batch_size, n_landmarks)

        # Calculate the mean Euclidean distance between the predicted and target landmarks
        distances = torch.norm(fine_preds - target, dim=-1)  # (batch_size, n_landmarks)
        distances_mean, distances_std = distances.mean(), distances.std()

        # Mask the distances with the loss mask
        distances_masked = distances * loss_mask
        distances_masked_mean, distances_masked_std = (
            distances_masked.mean(),
            distances_masked.std(),
        )

        metrics[f"fine_mean_distance_{mode}"] = distances_mean
        metrics[f"fine_std_distance_{mode}"] = distances_std
        metrics[f"fine_mean_distance_masked_{mode}"] = distances_masked_mean
        metrics[f"fine_std_distance_masked_{mode}"] = distances_masked_std

        # Calculate the magnitude of the offsets
        offsets = batch["offsets"]
        offsets_magnitude = torch.norm(offsets, dim=-1)
        offsets_magnitude_mean, offsets_magnitude_std = (
            offsets_magnitude.mean(),
            offsets_magnitude.std(),
        )

        # Mask the offsets with the loss mask
        offsets_masked = offsets_magnitude * loss_mask
        offsets_masked_mean, offsets_masked_std = (
            offsets_masked.mean(),
            offsets_masked.std(),
        )

        metrics[f"offsets_magnitude_mean_{mode}"] = offsets_magnitude_mean
        metrics[f"offsets_magnitude_std_{mode}"] = offsets_magnitude_std
        metrics[f"offsets_magnitude_masked_mean_{mode}"] = offsets_masked_mean
        metrics[f"offsets_magnitude_masked_std_{mode}"] = offsets_masked_std

        # Calculate mean Euclidian distance grouped by landmark type
        for i, landmark_type in enumerate(target_indices.unique()):
            landmark_mask = target_indices == landmark_type
            landmark_mask = landmark_mask * loss_mask
            distances_landmark = distances[landmark_mask]
            distances_landmark_mean = distances_landmark.mean()
            metrics[f"fine_mean_distance_{landmark_type.item()}_{mode}"] = (
                distances_landmark_mean
            )

        return metrics


class PatchTransformer(nn.Module):
    """Enrich the coarse features with patch features extracted with a simple CNN
    regressor, then refine the predictions using a transformer."""

    def __init__(
        self,
        n_landmarks: int,
        n_verts: int,
        patch_size: int,
        poi_feature_l: int,
        patch_feature_l: int,
        coord_embedding_l: int,
        poi_embedding_l: int,
        vert_embedding_l: int,
        loss_fn: str,
        project_gt: bool = False,
        warmup_epochs: int = -1,
        mlp_dim: int = 1024,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        lr: float = 1e-5,
    ):
        super().__init__()

        self.patch_feature_extractor = PatchExtractor(
            patch_size=patch_size,
            feature_extraction_model=Regressor(
                in_shape=(1, patch_size, patch_size, patch_size),
                out_shape=(patch_feature_l,),
                channels=(8, 16, 32),
                strides=(2, 2, 2),
                kernel_size=3,
            ),
        )

        self.refinement_module = PoiTransformer(
            poi_feature_l=poi_feature_l + patch_feature_l,
            coord_embedding_l=coord_embedding_l,
            poi_embedding_l=poi_embedding_l,
            vert_embedding_l=vert_embedding_l,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            n_landmarks=n_landmarks,
            n_verts=n_verts,
            dropout_rate=dropout,
        )

        self.project_gt = project_gt

        self.loss_fn = get_loss_fn(loss_fn)

        self.lr = lr

        self.warmup_epochs = warmup_epochs

    def forward(self, batch):
        coarse_preds = batch["coarse_preds"]
        poi_indices = batch["poi_list_idx"]
        vertebra_indices = batch["vert_list_idx"]
        poi_features = batch["coarse_features"]

        if (
            self.warmup_epochs > 0
            and self.training
            and self.current_epoch < self.warmup_epochs
        ):
            # Use a weighted average of ground truth and coarse predictions in the warmup phase
            coarse_preds = (
                self.current_epoch / self.warmup_epochs * coarse_preds
                + (1 - self.current_epoch / self.warmup_epochs) * batch["target"]
            )

        # Cast the coarse predictions to long to use them as indices
        coarse_preds = coarse_preds.detach().long()

        patches = self.patch_feature_extractor(batch["input"], coarse_preds)

        # Concatenate the patch features with the coarse features
        poi_features = torch.cat([poi_features, patches], dim=-1)

        offsets = self.refinement_module(
            coarse_preds, poi_indices, vertebra_indices, poi_features
        )
        batch["offsets"] = offsets
        batch["refined_preds"] = coarse_preds + offsets

        return batch

    def calculate_loss(self, batch):
        target = batch["target"]
        if self.project_gt:
            # Project targets to surface
            surface = batch["surface"]
            target, _ = surface_project_coords(target, surface)

        return self.loss_fn(batch["refined_preds"], target, batch["loss_mask"])

    def calculate_metrics(self, batch, mode):
        metrics = {}

        loss_mask = batch["loss_mask"]  # (batch_size, n_landmarks)
        fine_preds = batch["refined_preds"]  # (batch_size, n_landmarks, 3)
        target = batch["target"]  # (batch_size, n_landmarks, 3)
        target_indices = batch["target_indices"]  # (batch_size, n_landmarks)

        if self.project_gt:
            target = batch["target"]
            # Project targets to surface
            surface = batch["surface"]
            target, projection_dist = surface_project_coords(target, surface)

            metrics[f"fine_projection_dist_{mode}"] = projection_dist.mean()

        # Calculate the mean Euclidean distance between the predicted and target landmarks
        distances = torch.norm(fine_preds - target, dim=-1)  # (batch_size, n_landmarks)
        distances_mean, distances_std = distances.mean(), distances.std()

        # Mask the distances with the loss mask
        distances_masked = distances[loss_mask]
        distances_masked_mean, distances_masked_std = (
            distances_masked.mean(),
            distances_masked.std(),
        )

        metrics[f"fine_mean_distance_{mode}"] = distances_mean
        metrics[f"fine_std_distance_{mode}"] = distances_std
        metrics[f"fine_mean_distance_masked_{mode}"] = distances_masked_mean
        metrics[f"fine_std_distance_masked_{mode}"] = distances_masked_std

        # Calculate the magnitude of the offsets
        offsets = batch["offsets"]
        offsets_magnitude = torch.norm(offsets, dim=-1)
        offsets_magnitude_mean, offsets_magnitude_std = (
            offsets_magnitude.mean(),
            offsets_magnitude.std(),
        )

        # Mask the offsets with the loss mask
        offsets_masked = offsets_magnitude * loss_mask
        offsets_masked_mean, offsets_masked_std = (
            offsets_masked.mean(),
            offsets_masked.std(),
        )

        metrics[f"offsets_magnitude_mean_{mode}"] = offsets_magnitude_mean
        metrics[f"offsets_magnitude_std_{mode}"] = offsets_magnitude_std
        metrics[f"offsets_magnitude_masked_mean_{mode}"] = offsets_masked_mean
        metrics[f"offsets_magnitude_masked_std_{mode}"] = offsets_masked_std

        # Calculate mean Euclidian distance grouped by landmark type
        for i, landmark_type in enumerate(target_indices.unique()):
            landmark_mask = target_indices == landmark_type
            landmark_mask = landmark_mask * loss_mask
            distances_landmark = distances[landmark_mask]
            distances_landmark_mean = distances_landmark.mean()
            metrics[f"fine_mean_distance_{landmark_type.item()}_{mode}"] = (
                distances_landmark_mean
            )

        return metrics
