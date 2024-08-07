import nibabel as nib
import numpy as np
import torch
from BIDS import NII, POI


def np_to_bids_nii(array: np.ndarray) -> NII:
    """Converts a numpy array to a BIDS NII object."""
    # NiBabel expects the orientation to be RAS+ (right, anterior, superior, plus),
    # we have LAS+ (left, posterior, superior, plus) so we need to flip along the second axis
    affine = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    nifty1img = nib.Nifti1Image(array, affine)
    return NII(nifty1img)


def one_hot_encode_batch(batch_tensor: torch.Tensor) -> torch.Tensor:
    """One hot encodes a batch of labels."""
    batch_tensor = batch_tensor.squeeze(1).long()

    num_classes = 11
    batch_tensor = batch_tensor - 40
    batch_tensor[batch_tensor < 0] = 0

    batch_tensor = (
        torch.nn.functional.one_hot(batch_tensor, num_classes=num_classes)
        .permute(0, 4, 1, 2, 3)
        .float()
    )

    return batch_tensor


def surface_project_coords(coordinates, surface):
    unbatched = len(coordinates.shape) == 2
    if unbatched:
        coordinates = coordinates.detach().clone().unsqueeze(0)
        surface = surface.detach().clone().unsqueeze(0)
    B, N, _ = coordinates.shape
    device = coordinates.device
    surface_projected_targets = torch.zeros_like(coordinates, dtype=torch.int64)
    surface_projection_dist = torch.zeros(B, N, dtype=torch.float32)

    for b in range(B):
        for i in range(N):
            coord = coordinates[b, i, :]
            # Create a mask of all 'true' surface points for this batch
            surface_points_indices = surface[b].squeeze(0).nonzero(as_tuple=False)
            # Calculate Euclidean distances from the current point to all surface points
            distances = torch.sqrt(((surface_points_indices - coord) ** 2).sum(dim=1))
            # Find the index of the closest surface point
            min_dist_index = torch.argmin(distances)
            closest_point = surface_points_indices[min_dist_index]
            surface_projected_targets[b, i, :] = closest_point
            surface_projection_dist[b, i] = distances[min_dist_index]
    return surface_projected_targets.to(device), surface_projection_dist.to(device)


# POI Visualization
# Define some useful utility functions
def get_dd_ctd(dd, poi_list=None):
    ctd = {}
    vertebra = dd["vertebra"]

    for poi_coords, poi_idx in zip(dd["target"], dd["target_indices"]):
        coords = (poi_coords[0].item(), poi_coords[1].item(), poi_coords[2].item())
        if poi_list is None or poi_idx in poi_list:
            ctd[vertebra, poi_idx.item()] = coords

    ctd = POI(
        centroids=ctd, orientation=("L", "A", "S"), zoom=(1, 1, 1), shape=(128, 128, 96)
    )
    return ctd


def get_ctd(target, target_indices, vertebra, poi_list):
    ctd = {}
    for poi_coords, poi_idx in zip(target, target_indices):
        coords = (poi_coords[0].item(), poi_coords[1].item(), poi_coords[2].item())
        if poi_list is None or poi_idx in poi_list:
            ctd[vertebra, poi_idx.item()] = coords

    ctd = POI(
        centroids=ctd, orientation=("L", "A", "S"), zoom=(1, 1, 1), shape=(128, 128, 96)
    )
    return ctd


def get_vert_msk_nii(dd):
    vertebra = dd["vertebra"]
    msk = dd["input"].squeeze(0)
    return vertseg_to_vert_msk_nii(vertebra, msk)


def vertseg_to_vert_msk_nii(vertebra, msk):
    vert_msk = (msk != 0) * vertebra
    vert_msk_nii = np_to_bids_nii(vert_msk.numpy().astype(np.int32))
    vert_msk_nii.seg = True
    return vert_msk_nii


def get_vertseg_nii(dd):
    vertseg = dd["input"].squeeze(0)
    vertseg_nii = np_to_bids_nii(vertseg.numpy().astype(np.int32))
    vertseg_nii.seg = True
    return vertseg_nii


def get_vert_points(dd):
    msk = dd["input"].squeeze(0)
    vert_points = torch.where(msk)
    vert_points = torch.stack(vert_points, dim=1)
    return vert_points


def get_target_entry_points(dd):
    ctd = get_ctd(dd)
    vertebra = dd["vertebra"]
    p_90 = torch.tensor(ctd[vertebra, 90])
    p_92 = torch.tensor(ctd[vertebra, 92])

    p_91 = torch.tensor(ctd[vertebra, 91])
    p_93 = torch.tensor(ctd[vertebra, 93])

    return p_90, p_92, p_91, p_93


def tensor_to_ctd(
    t,
    vertebra,
    origin,
    rotation,
    idx_list=None,
    shape=(128, 128, 96),
    zoom=(1, 1, 1),
    offset=(0, 0, 0),
):
    ctd = {}
    for i, coords in enumerate(t):
        coords = coords.float() - torch.tensor(offset)
        coords = (coords[0].item(), coords[1].item(), coords[2].item())
        if idx_list is None:
            ctd[vertebra, i] = coords
        elif i < len(idx_list):
            ctd[vertebra, idx_list[i]] = coords

    ctd = POI(
        centroids=ctd,
        orientation=("L", "A", "S"),
        zoom=zoom,
        shape=shape,
        origin=origin,
        rotation=rotation,
    )
    return ctd
