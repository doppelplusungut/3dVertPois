import os
from os import PathLike
from typing import Callable

import numpy as np
import torch
from BIDS import NII, POI
from BIDS.bids_files import Subject_Container
from numpy import ndarray
from scipy.ndimage import center_of_mass, shift
from skimage.morphology import binary_erosion


def compute_zoom(original_shape, original_zoom, target_shape):
    # Compute the zoom factor to resize the image to the target shape
    zoom = np.array(original_shape) / np.array(target_shape) * np.array(original_zoom)
    return (zoom[0], zoom[1], zoom[2])


def get_gt_pois(poi, vertebra, poi_indices):
    """Converts the POI coordinates to a tensor.

    Args:
        poi (POI): The POI coordinates.
        vertebra (int): The vertebra number.

    Returns:
        torch.Tensor: The POI coordinates as a tensor.
    """
    coords = [
        (
            np.array((-1, -1, -1))
            if not (vertebra, p_idx) in poi.keys()
            else np.array(poi.centroids[vertebra, p_idx])
        )
        for p_idx in poi_indices
    ]

    # Stack the coordinates
    coords = np.stack(coords)

    # Change type of coords to float
    coords = coords.astype(np.float32)  # Shape: (n_pois, 3)

    # Mark the missing pois
    missing_poi_list_idx = np.all(coords == -1, axis=1)  # Shape: (n_pois,)

    # Get the indices of missing pois
    missing_pois = np.array(
        [poi_idx for i, poi_idx in enumerate(poi_indices) if missing_poi_list_idx[i]]
    )

    return torch.from_numpy(coords), torch.from_numpy(missing_pois)


def get_gt_hm(coords, target_shape, ref_heatmap, ref_hm_center):
    # Compute target heatmap from coordinates
    n_pois, _ = coords.shape
    heatmaps = np.zeros(shape=(n_pois, *target_shape))

    for poi_idx in range(n_pois):
        coord = coords[poi_idx]
        # Heatmaps are all an exponential decay from the center, just shifted to the correct location
        heatmap = shift(ref_heatmap, coord - ref_hm_center)
        # Normalize the heatmap
        heatmap = heatmap / heatmap.sum()
        heatmaps[poi_idx] = heatmap

    return heatmaps


def get_gt_hm_torch(coords, target_shape, patch=9, sigma=1):
    dens = get_density(1, (patch, patch, patch))

    heatmaps = [embed_patch(dens, target_shape, tuple(coord)) for coord in coords]
    heatmaps = [heatmap / heatmap.sum() for heatmap in heatmaps]
    heatmaps = torch.stack(heatmaps)
    return heatmaps


def get_gt_hm_torch_batch(coords_batch, target_shape, patch=9, sigma=1):
    batch_size = coords_batch.size(0)
    heatmaps = []

    for b in range(batch_size):
        hm = get_gt_hm_torch(coords_batch[b, :, :], target_shape, patch, sigma)
        heatmaps.append(hm)

    heatmaps = torch.stack(heatmaps)
    return heatmaps


def get_density(sigma, shape):
    """Returns a tensor of the specified shape containing the discretely sampled density
    of a Gaussian distribution, centered at the middle of the tensor, with sigma as the
    standard deviation parameter.

    Parameters:
    - sigma: The standard deviation of the Gaussian distribution.
    - shape: The shape of the output tensor (2D or 3D).

    Returns:
    - A tensor of shape 'shape' with the Gaussian density sampled across its volume.
    """
    # Create a grid of indices
    indices = [torch.arange(s, dtype=torch.float32) for s in shape]
    grid = torch.meshgrid(indices, indexing="ij")

    # Calculate the center
    center = torch.tensor([int((s - 1) / 2) for s in shape], dtype=torch.float32)

    # Calculate squared distances from the center
    squared_distances = sum([(g - c) ** 2 for g, c in zip(grid, center)])

    # Calculate the Gaussian distribution
    # distribution = (1 / (sigma * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(-squared_distances / (2 * sigma ** 2))
    distribution = torch.exp(-squared_distances / (2 * sigma**2))
    return distribution


def embed_patch(patch, target_shape, center):
    """Embeds a 3D patch into a tensor of target shape with the patch centered at a
    specified index.

    Parameters:
    - patch: Tensor of shape (p, p, p).
    - target_shape: Tuple of (H, W, D) indicating the shape of the target tensor.
    - center: Tuple of (x, y, z) indicating the center index in the target tensor.

    Returns:
    - A tensor of shape target_shape with the patch embedded and the rest filled with zeros.
    """
    # Convert center to integers
    center = [int(c) for c in center]

    # Create the target tensor filled with zeros
    target_tensor = torch.zeros(target_shape)

    p = patch.shape[0]  # Assuming patch is a cubic shape (p, p, p)
    # Calculate the start and end indices for embedding the patch in the target tensor
    start_indices = [center[i] - p // 2 for i in range(3)]
    end_indices = [center[i] + (p + 1) // 2 for i in range(3)]

    # Calculate the actual start and end indices in the patch to be used
    patch_start = [0 if start_indices[i] >= 0 else -start_indices[i] for i in range(3)]
    patch_end = [
        (
            p
            if end_indices[i] <= target_shape[i]
            else p - (end_indices[i] - target_shape[i])
        )
        for i in range(3)
    ]

    # Adjust start and end indices to be within target tensor bounds
    start_indices = [max(0, start) for start in start_indices]
    end_indices = [min(target_shape[i], end_indices[i]) for i in range(3)]

    # Embed the patch into the target tensor
    target_tensor[
        start_indices[0] : end_indices[0],
        start_indices[1] : end_indices[1],
        start_indices[2] : end_indices[2],
    ] = patch[
        patch_start[0] : patch_end[0],
        patch_start[1] : patch_end[1],
        patch_start[2] : patch_end[2],
    ]

    return target_tensor


def compute_surface(msk: torch.tensor, iterations=1) -> torch.tensor:
    """Computes the surface of the vertebra.

    Args:
        msk (numpy.ndarray): The segmentation mask.
        vertebra (int): The vertebra number.

    Returns:
        torch.Tensor: The surface of the vertebra.
    """
    surface = msk.numpy()

    eroded = surface.copy()
    for _ in range(iterations):
        eroded = binary_erosion(eroded)

    surface[eroded] = 0

    return torch.from_numpy(surface)


def one_hot_encode_3d(
    array: np.ndarray, subreg_ids: list[int] = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
) -> np.ndarray:
    """Convert a 3D integer numpy array to one-hot encoding with the channel as the
    first dimension.

    Parameters:
    - array: A 3D numpy array with integer values.
    - class_values: A list of unique integer values representing the classes.

    Returns:
    - A 4D numpy array with one-hot encoding along the first dimension.
    """
    # Create a 4D array of zeros with the first dimension being the number of classes
    one_hot_encoded = np.zeros((len(subreg_ids),) + array.shape, dtype=np.int16)

    # Iterate over each class and encode it in the corresponding channel
    for i, value in enumerate(subreg_ids):
        one_hot_encoded[i] = array == value

    return one_hot_encoded


def apply_dictionary_transform(
    transform: callable, im: ndarray, subreg: ndarray, vertseg: ndarray, poi_hm: ndarray
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """Apply a random affine transformation to the input and target.

    Args:
    - im: The input image.
    - subreg: The subregional segmentation.
    - vertseg: The vertebral segmentation.
    - poi_hm: The heatmap of the points of interest.
    """

    # Add channel dimension to the input
    im = np.expand_dims(im, axis=0)
    subreg = np.expand_dims(subreg, axis=0)
    vertseg = np.expand_dims(vertseg, axis=0)

    # Create a dictionary with the input and target
    data_dict = {"im": im, "subreg": subreg, "vertseg": vertseg, "target": poi_hm}

    transformed_data_dict = transform(data_dict)

    # Convert back to numpy
    im = transformed_data_dict["im"]
    subreg = transformed_data_dict["subreg"]
    vertseg = transformed_data_dict["vertseg"]

    # Remove channel dimension
    im = np.squeeze(im, axis=0)
    subreg = np.squeeze(subreg, axis=0)
    vertseg = np.squeeze(vertseg, axis=0)

    return im, subreg, vertseg, transformed_data_dict["target"]


def get_subreg_com(
    subreg: ndarray, subregs=[41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
) -> ndarray:
    """Compute the center of mass of the subregional segmentation.

    Args:
    - subreg: The subregional segmentation.

    Returns:
    - The center of mass for each subregion.
    """
    idx = []
    coms = []
    for subreg_id in subregs:
        msk = subreg == subreg_id
        if msk.any():
            com = center_of_mass(msk)
            coms.append(com)
            idx.append(subreg_id)
        else:
            coms.append((0, 0, 0))
            idx.append(0)

    idx = np.array(idx)  # Shape: (n_subregs,)
    coms = np.array(coms).astype(np.float32)  # Shape: (n_subregs, 3)

    return idx, coms


def pad_array_to_shape(arr, target_shape):
    """Pads the input array arr to the target_shape. The original array is centered
    within the new shape.

    Parameters:
    - arr: numpy array of shape (H, W, D)
    - target_shape: tuple of the target shape (H', W', D')

    Returns:
    - Padded numpy array of shape (H', W', D')
    """
    # Calculate the padding needed for each dimension
    pad_h = (target_shape[0] - arr.shape[0]) // 2
    pad_w = (target_shape[1] - arr.shape[1]) // 2
    pad_d = (target_shape[2] - arr.shape[2]) // 2

    # Handle odd differences by adding an extra padding at the end if necessary
    pad_h2 = pad_h + (target_shape[0] - arr.shape[0]) % 2
    pad_w2 = pad_w + (target_shape[1] - arr.shape[1]) % 2
    pad_d2 = pad_d + (target_shape[2] - arr.shape[2]) % 2

    # Apply padding
    padded_arr = np.pad(
        arr, ((pad_h, pad_h2), (pad_w, pad_w2), (pad_d, pad_d2)), mode="constant"
    )

    offset = (pad_h, pad_w, pad_d)

    return padded_arr, offset


def create_coordinate_tensor(shape):
    # Generate a grid of coordinates along each dimension
    indices = [torch.arange(s, dtype=torch.float32) for s in shape]
    grid = torch.meshgrid(indices, indexing="ij")

    # Stack the coordinate grids along a new dimension to get the final coordinate tensor
    coords = torch.stack(grid, dim=0).float()

    return coords


def heatmaps_to_coords(pred_heatmaps):
    """Compute 3D coordinates from a batch of heatmaps, by taking the weighted average
    of the heatmap coordinates.

    Args:
        pred_heatmaps (torch.Tensor): Batch of heatmaps of shape (batch_size, n_pois, height, width, depth).

    Returns:
        torch.Tensor: Batch of 3D coordinates of shape (batch_size, n_pois, 3).
    """
    # Extract dimensions
    device = pred_heatmaps.device
    batch_size, n_pois, _, _, _ = pred_heatmaps.shape

    # Reshape pred_heatmaps and heatmap_coords for element-wise multiplication
    pred_heatmaps_reshaped = pred_heatmaps.view(
        batch_size, n_pois, -1
    )  # Shape: (batch_size, n_pois, H*W*D)
    heatmap_coords = create_coordinate_tensor(pred_heatmaps.shape[2:]).to(device)
    heatmap_coords_reshaped = heatmap_coords.view(3, -1)  # Shape: (3, H*W*D)

    heatmap_coords_reshaped = heatmap_coords_reshaped.to(device)

    # Element-wise multiplication and sum along the last dimension
    weighted_coords = torch.sum(
        pred_heatmaps_reshaped.unsqueeze(-1) * heatmap_coords_reshaped.t(), dim=2
    )  # Shape: (batch_size, n_pois, 3)

    # Reshape the result to (batch_size, n_pois, 3)
    coords = weighted_coords.view(batch_size, n_pois, 3)

    return coords.to(device)


def get_implants_poi(container) -> POI:
    poi_query = container.new_query(flatten=True)
    poi_query.filter_format("poi")
    poi_query.filter("desc", "local")
    poi_candidate = poi_query.candidates[0]

    poi = poi_candidate.open_ctd()
    return poi


def get_gruber_poi(container) -> POI:
    poi_query = container.new_query(flatten=True)
    poi_query.filter_format("poi")
    poi_query.filter("source", "gruber")
    poi_candidate = poi_query.candidates[0]

    poi = poi_candidate.open_ctd()
    return poi


def get_gruber_registration_poi(container):
    poi_query = container.new_query(flatten=True)
    poi_query.filter_format("poi")
    poi_query.filter("source", "registered")
    poi_query.filter_filetype(".json")

    registration_ctds = [POI.load(poi) for poi in poi_query.candidates]

    # Check whether zoom, shape and direction coincide
    for i in range(1, len(registration_ctds)):
        if not registration_ctds[0].zoom == registration_ctds[i].zoom:
            print("Zoom does not match")
        if not registration_ctds[0].shape == registration_ctds[i].shape:
            print("Shape does not match")
        if not registration_ctds[0].orientation == registration_ctds[i].orientation:
            print("Direction does not match")

    # Get the keys that are present in all POIs
    keys = set(registration_ctds[0].keys())
    for ctd in registration_ctds:
        keys = keys.intersection(set(ctd.keys()))
    keys = list(keys)

    ctd = {}
    for key in keys:
        #
        ctd[key] = tuple(
            np.array([reg_ctd[key] for reg_ctd in registration_ctds]).mean(axis=0)
        )

    # Sort the new ctd by keys
    ctd = dict(sorted(ctd.items()))
    new_poi = POI(
        centroids=ctd,
        orientation=registration_ctds[0].orientation,
        zoom=registration_ctds[0].zoom,
        shape=registration_ctds[0].shape,
    )

    return new_poi


def get_ct(container) -> NII:
    ct_query = container.new_query(flatten=True)
    ct_query.filter_format("ct")
    ct_query.filter_filetype("nii.gz")  # only nifti files
    ct_candidate = ct_query.candidates[0]

    ct = ct_candidate.open_nii()
    return ct


def get_subreg(container) -> NII:
    subreg_query = container.new_query(flatten=True)
    subreg_query.filter_format("msk")
    subreg_query.filter_filetype("nii.gz")  # only nifti files
    subreg_query.filter("seg", "subreg")
    subreg_candidate = subreg_query.candidates[0]

    subreg = subreg_candidate.open_nii()
    return subreg


def get_vertseg(container) -> NII:
    vertseg_query = container.new_query(flatten=True)
    vertseg_query.filter_format("msk")
    vertseg_query.filter_filetype("nii.gz")  # only nifti files
    vertseg_query.filter("seg", "vert")
    vertseg_candidate = vertseg_query.candidates[0]

    vertseg = vertseg_candidate.open_nii()
    return vertseg


def get_files(
    container,
    get_poi: callable,
    get_ct: callable,
    get_subreg: callable,
    get_vertseg: callable,
) -> tuple[POI, NII, NII, NII]:
    return (
        get_poi(container),
        get_ct(container),
        get_subreg(container),
        get_vertseg(container),
    )


def get_bounding_box(mask, vert, margin=5):
    """Get the bounding box of a given vertebra in a mask.

    Args:
        mask (numpy.ndarray): The mask to search for the vertex.
        vert (int): The vertebra to search for in the mask.
        margin (int, optional): The margin to add to the bounding box. Defaults to 2.

    Returns:
        tuple: A tuple containing the minimum and maximum values for the x, y, and z axes of the bounding box.
    """
    indices = np.where(mask == vert)
    x_min = np.min(indices[0]) - margin
    x_max = np.max(indices[0]) + margin
    y_min = np.min(indices[1]) - margin
    y_max = np.max(indices[1]) + margin
    z_min = np.min(indices[2]) - margin
    z_max = np.max(indices[2]) + margin

    # Make sure the bounding box is within the mask
    x_min = max(0, x_min)
    x_max = min(mask.shape[0], x_max)
    y_min = max(0, y_min)
    y_max = min(mask.shape[1], y_max)
    z_min = max(0, z_min)
    z_max = min(mask.shape[2], z_max)

    return x_min, x_max, y_min, y_max, z_min, z_max


def process_container(
    subject,
    container,
    save_path: PathLike,
    rescale_zoom: tuple | None,
    get_files: Callable[[Subject_Container], tuple[POI, NII, NII, NII]],
):
    poi, ct, subreg, vertseg = get_files(container)
    ct.reorient_(("L", "A", "S"))
    subreg.reorient_(("L", "A", "S"))
    vertseg.reorient_(("L", "A", "S"))
    poi.reorient_centroids_to_(ct)

    vertebrae = set([key[0] for key in poi.keys()])
    vertseg_arr = vertseg.get_array()

    summary = []
    for vert in vertebrae:
        if vert in vertseg_arr:
            x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box(
                vertseg_arr, vert
            )
            ct_path = os.path.join(save_path, subject, str(vert), "ct.nii.gz")
            subreg_path = os.path.join(save_path, subject, str(vert), "subreg.nii.gz")
            vertseg_path = os.path.join(save_path, subject, str(vert), "vertseg.nii.gz")
            poi_path = os.path.join(save_path, subject, str(vert), "poi.json")
            # poi_det_path = os.path.join(save_path, subject, str(vert), 'poi_det.json')

            if not os.path.exists(os.path.join(save_path, subject, str(vert))):
                os.makedirs(os.path.join(save_path, subject, str(vert)))

            ct_cropped = ct.apply_crop_slice(
                ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
            )
            subreg_cropped = subreg.apply_crop_slice(
                ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
            )
            vertseg_cropped = vertseg.apply_crop_slice(
                ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
            )
            poi_cropped = poi.crop_centroids(
                o_shift=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
            )
            # poi_det.crop_centroids(o_shift = (slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))).save(poi_det_path)

            if rescale_zoom:
                ct_cropped.rescale_(rescale_zoom)
                subreg_cropped.rescale_(rescale_zoom)
                vertseg_cropped.rescale_(rescale_zoom)
                poi_cropped.rescale_(rescale_zoom)

            ct_cropped.save(ct_path, verbose=False)
            subreg_cropped.save(subreg_path, verbose=False)
            vertseg_cropped.save(vertseg_path, verbose=False)
            poi_cropped.save(poi_path, verbose=False)

            summary.append(
                {
                    "subject": subject,
                    "vertebra": vert,
                    "file_dir": os.path.join(save_path, subject, str(vert)),
                    # 'ct_nii_path': ct_path,
                    # 'subreg_nii_path': subreg_path,
                    # 'vertseg_nii_path': vertseg_path,
                    # 'poi_json_path': poi_path,
                }
            )

        else:
            print(f"Vertebra {vert} has no segmentation for subject {subject}")

    return summary
