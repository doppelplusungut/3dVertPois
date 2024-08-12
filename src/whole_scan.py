"""Inference Pipeline:
Given a path to a vert and subreg segmentation mask, model and data module, this pipeline will:
1. Load the vert and subreg mask
2. Create vertebra-wise cutouts and a master_df in a temporary directory
3. Reorient and rescale the cutouts to (1,1,1) mm resolution
4. Pad the cutouts to a fixed size
5. Arrange the cutouts into a batch
6. Pass the batch through the model
7. Extract the predicted landmarks from the model output
8. Revert the predicted landmarks to the original space (remove padding, rescale, reorient, remove margin, add offset)
9. Save the predicted landmarks to a BIDS POI file (.json)
10. Delete the temporary directory
"""

import ast
import json
import os

import pandas as pd
import torch
from BIDS import NII, POI, BIDS_Global_info
from torch.utils.data import Dataset

import eval as ev
from prepare_data import get_bounding_box
from utils.dataloading_utils import compute_surface, pad_array_to_shape
from utils.misc import surface_project_coords

dm_path = "/home/daniel/Projects/gruber/surface/all_pois/freeze/SA-DenseNet-PatchTransformer/version_4/data_module_params.json"
model_path = "/home/daniel/Projects/gruber/surface/all_pois/freeze/SA-DenseNet-PatchTransformer/version_4/checkpoints/sad-pt-epoch=44-fine_mean_distance_val=2.33.ckpt"


def get_subreg(container):
    subreg_query = container.new_query(flatten=True)
    subreg_query.filter_format("msk")
    subreg_query.filter_filetype("nii.gz")  # only nifti files
    subreg_query.filter("seg", "subreg")
    subreg_candidate = subreg_query.candidates[0]
    return str(subreg_candidate.file["nii.gz"])


def get_vertseg(container):
    vertseg_query = container.new_query(flatten=True)
    vertseg_query.filter_format("msk")
    vertseg_query.filter_filetype("nii.gz")  # only nifti files
    vertseg_query.filter("seg", "vert")
    vertseg_candidate = vertseg_query.candidates[0]
    return str(vertseg_candidate.file["nii.gz"])


def combine_centroids(data_list):
    # Extract the first dictionary for comparison
    first_entry = data_list[0]

    # Define the expected values for comparison
    expected_subject = first_entry["subject"]
    expected_shape = first_entry["original_shape"]
    expected_zoom = first_entry["original_zoom"]
    expected_orientation = first_entry["original_orientation"]

    # Initialize a defaultdict for combining centroids
    combined_centroids = {}

    # Iterate through each entry in the list
    for entry in data_list:
        # Assert that subject, shape, zoom, and orientation match the expected values
        assert entry["subject"] == expected_subject, "Subjects do not match."
        assert (
            entry["original_shape"] == expected_shape
        ), "Original shapes do not match."
        assert entry["original_zoom"] == expected_zoom, "Original zooms do not match."
        assert (
            entry["original_orientation"] == expected_orientation
        ), "Original orientations do not match."

        # Combine the centroids
        for v_idx, p_idx, c in entry["centroids"].items():
            combined_centroids[v_idx, p_idx] = c

    # Convert combined_centroids to a regular dict
    combined_centroids = dict(combined_centroids)

    # Return the common attributes and the combined centroids
    poi_file = POI(
        centroids=combined_centroids,
        orientation=expected_orientation,
        zoom=expected_zoom,
        shape=expected_shape,
    )

    return expected_subject, poi_file


class GruberInferenceDataset(Dataset):
    def __init__(
        self,
        master_df,
        input_shape,
        include_vert_list,
        poi_indices=[
            81,
            101,
            102,
            103,
            104,
            109,
            110,
            111,
            112,
            117,
            118,
            119,
            120,
            125,
            127,
            134,
            136,
            141,
            142,
            143,
            144,
            149,
            151,
        ],
    ):
        self.master_df = master_df
        self.input_shape = input_shape
        self.poi_indices = torch.tensor(poi_indices)
        self.poi_idx_to_list_idx = {poi: idx for idx, poi in enumerate(poi_indices)}
        self.vert_idx_to_list_idx = {
            vert: idx for idx, vert in enumerate(include_vert_list)
        }

    def __len__(self):
        return len(self.master_df)

    def __getitem__(self, index):
        data_dict = {}

        # Read the row from the master dataframe
        row = self.master_df.iloc[index]
        vertebra = row["vert"]
        vert_path = row["vert_path"]
        subreg_path = row["subreg_path"]
        x_min = row["x_min"]
        y_min = row["y_min"]
        z_min = row["z_min"]
        original_orientation = row["original_orientation"]
        original_zoom = row["original_zoom"]
        original_shape = row["original_shape"]
        subject = row["subject"]

        subreg = NII.load(subreg_path, seg=True)
        vertseg = NII.load(vert_path, seg=True)

        assert subreg.shape == vertseg.shape
        assert subreg.orientation == vertseg.orientation
        assert subreg.orientation == ("L", "A", "S")
        assert subreg.zoom == vertseg.zoom
        assert subreg.zoom == (1, 1, 1)

        subreg = subreg.get_array()
        vertseg = vertseg.get_array()
        mask = vertseg == vertebra

        # ct = ct * mask
        subreg = subreg * mask

        subreg, offset = pad_array_to_shape(subreg, self.input_shape)
        vertseg, _ = pad_array_to_shape(vertseg, self.input_shape)

        # Convert subreg and vertseg to tensors
        subreg = torch.from_numpy(subreg.astype(float))
        vertseg = torch.from_numpy(vertseg.astype(float))

        # Add channel dimension
        subreg = subreg.unsqueeze(0)
        vertseg = vertseg.unsqueeze(0)

        # Uses default iterations of 1, must be changed if model was trained with more iterations ("thicker" surface)
        surface = compute_surface(subreg)

        data_dict["input"] = subreg
        data_dict["surface"] = surface
        data_dict["vertebra"] = vertebra
        data_dict["padding_offset"] = torch.tensor(offset).float()
        data_dict["poi_indices"] = self.poi_indices
        data_dict["poi_list_idx"] = torch.tensor(
            [self.poi_idx_to_list_idx[poi.item()] for poi in self.poi_indices]
        )
        data_dict["vert_list_idx"] = torch.tensor([self.vert_idx_to_list_idx[vertebra]])
        data_dict["cutout_offset"] = torch.tensor([x_min, y_min, z_min])
        data_dict["original_orientation"] = str(original_orientation)
        data_dict["original_zoom"] = original_zoom
        data_dict["original_shape"] = original_shape
        data_dict["subject"] = subject

        return data_dict


def predict(subject, vert_msk_path, subreg_msk_path, model_path, dm_path, save_dir):
    # Load the vert and subreg mask
    vert_msk = NII.load(vert_msk_path, seg=True)
    subreg_msk = NII.load(subreg_msk_path, seg=True)

    # Save the original orientation and zoom for later
    original_orientation = vert_msk.orientation
    original_zoom = vert_msk.zoom
    original_shape = vert_msk.shape

    # Load data module parameters
    dm_params = json.load(open(dm_path, "r"))
    input_shape = dm_params["input_shape"]
    vert_list = dm_params["include_vert_list"]
    poi_indices = dm_params["include_poi_list"]

    # Create temp directory
    temp_dir = "tmp/"
    os.makedirs(os.path.join(temp_dir, subject), exist_ok=True)

    # Get vertebrae that are both in the vert_list and in the vert mask
    msk_vert_list = vert_msk.unique()
    vertebrae = [v for v in vert_list if v in msk_vert_list]

    # Bring the masks to standard orientation. Zoom is applied AFTER cutting out the vertebrae
    vert_msk.reorient_(("L", "A", "S"))
    subreg_msk.reorient_(("L", "A", "S"))

    # Load the data array
    vertseg_arr = vert_msk.get_array()

    # Create vertebra-wise cutouts and a master_df in a temporary directory
    cutout_info = []
    for vert in vertebrae:
        # This uses the standard margin of 5 voxels around the vertebra in each direction. When the model is trained with a different margin, this should be adjusted!
        x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box(vertseg_arr, vert)

        subreg_path = os.path.join(temp_dir, subject, f"vert_{vert}-subreg.nii.gz")
        vert_path = os.path.join(temp_dir, subject, f"vert_{vert}-vertseg.nii.gz")

        subreg_cropped = subreg_msk.apply_crop_slice(
            ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
        )

        vert_cropped = vert_msk.apply_crop_slice(
            ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
        )

        # Reorient and rescale the cutouts to (1,1,1) mm resolution
        vert_cropped.rescale_((1, 1, 1))
        subreg_cropped.rescale_((1, 1, 1))
        subreg_cropped.save(subreg_path, verbose=False)
        vert_cropped.save(vert_path, verbose=False)

        # Save the slice indices as json to reconstruct the original POI file (there probably is a more BIDS-like approach to this)
        cutout_info.append(
            {
                "subject": subject,
                "vert": vert,
                "x_min": int(x_min),
                "x_max": int(x_max),
                "y_min": int(y_min),
                "y_max": int(y_max),
                "z_min": int(z_min),
                "z_max": int(z_max),
                "vert_path": vert_path,
                "subreg_path": subreg_path,
                "original_orientation": original_orientation,
                "original_zoom": original_zoom,
                "original_shape": original_shape,
            }
        )

    # Read the cutout info into a DataFrame
    master_df = pd.DataFrame(cutout_info)

    # Save the master_df to a csv file
    master_df_path = os.path.join(temp_dir, subject, "cutout_df.csv")
    master_df.to_csv(master_df_path, index=False)

    # Load data module parameters
    dm_params = json.load(open(dm_path, "r"))
    input_shape = dm_params["input_shape"]
    vert_list = dm_params["include_vert_list"]
    poi_indices = dm_params["include_poi_list"]

    ds = GruberInferenceDataset(
        master_df, input_shape=input_shape, include_vert_list=vert_list
    )

    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    model = ev.load_model_from_checkpoint(model_path)

    partial_centroids = []

    for batch in dl:
        batch = {
            k: v.to(model.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        batch = model(batch)
        refined_preds_batch = batch["refined_preds"]
        refined_preds_projected_batch, _ = surface_project_coords(
            refined_preds_batch, batch["surface"]
        )
        pred_coords = refined_preds_projected_batch.squeeze().detach().cpu().numpy()
        padding_offset = batch["padding_offset"].squeeze().detach().cpu().numpy()
        vertebra = batch["vertebra"].squeeze().detach().cpu().numpy()
        poi_indices = batch["poi_indices"].squeeze().detach().cpu().numpy()

        original_orientation = ast.literal_eval(batch["original_orientation"][0])
        original_zoom = (
            batch["original_zoom"][0][0].item(),
            batch["original_zoom"][1][0].item(),
            batch["original_zoom"][2][0].item(),
        )
        original_shape = (
            batch["original_shape"][0][0].item(),
            batch["original_shape"][1][0].item(),
            batch["original_shape"][2][0].item(),
        )

        subject = batch["subject"][0]

        cutout_offset = batch["cutout_offset"].squeeze().detach().cpu().numpy()

        unpadded_refined_preds_ctd = ev.np_to_ctd(
            pred_coords,
            vertebra=vertebra.item(),
            origin=None,
            rotation=None,
            idx_list=poi_indices,
            shape=input_shape,
            zoom=(1, 1, 1),
            offset=padding_offset,
        )

        # Rescale and reorient the predicted landmarks to the original space
        unpadded_refined_preds_ctd.rescale_(original_zoom)
        unpadded_refined_preds_ctd.reorient_(original_orientation)

        # Finally, add the cutout offset to the predicted landmarks
        new_centroids = {}
        for v, p_idx, c in unpadded_refined_preds_ctd.centroids.items():
            new_coords = c + cutout_offset
            new_centroids[(v, p_idx)] = (new_coords[0], new_coords[1], new_coords[2])

        unpadded_refined_preds_ctd.centroids = new_centroids

        partial_centroids.append(
            {
                "subject": subject,
                "original_shape": original_shape,
                "original_zoom": original_zoom,
                "original_orientation": original_orientation,
                "centroids": unpadded_refined_preds_ctd.centroids,
            }
        )

    sub, pois = combine_centroids(partial_centroids)

    os.makedirs(os.path.join(save_dir, sub), exist_ok=True)

    pois.save(os.path.join(save_dir, sub, "poi_predicted.json"))

    # Clear the temporary directory
    os.system(f"rm -r {temp_dir}")


if __name__ == "__main__":
    bgi = BIDS_Global_info(
        datasets=["/home/daniel/MEGA downloads/dataset-gruber"],
        parents=["derivatives_seg_new"],
    )

    for sub, container in bgi.enumerate_subjects():
        vert_msk_path = get_vertseg(container)
        subreg_msk_path = get_subreg(container)
        predict(
            sub,
            vert_msk_path,
            subreg_msk_path,
            model_path,
            dm_path,
            "/home/daniel/MEGAsync/Uni",
        )
