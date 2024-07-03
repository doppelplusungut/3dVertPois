import argparse
import os
from os import PathLike
from typing import Callable
from functools import partial
import json

import pandas as pd
import numpy as np
from pqdm.processes import pqdm

from BIDS import NII, POI, BIDS_Global_info
from BIDS.bids_files import Subject_Container

def get_implants_poi(container) -> POI:
    poi_query = container.new_query(flatten=True)
    poi_query.filter_format('poi')
    poi_query.filter('desc', 'local')
    poi_candidate = poi_query.candidates[0]
    
    poi = poi_candidate.open_ctd()
    return poi

def get_gruber_poi(container) -> POI:
    poi_query = container.new_query(flatten=True)
    poi_query.filter_format('poi')
    poi_query.filter('source', 'gruber')
    poi_candidate = poi_query.candidates[0]
    
    poi = poi_candidate.open_ctd()
    return poi

def get_gruber_registration_poi(container):
    poi_query = container.new_query(flatten=True)
    poi_query.filter_format('poi')
    poi_query.filter('source', 'registered')
    poi_query.filter_filetype('.json')

    registration_ctds = [POI.load(poi) for poi in poi_query.candidates]

    #Check whether zoom, shape and direction coincide
    for i in range(1, len(registration_ctds)):
        if not registration_ctds[0].zoom == registration_ctds[i].zoom:
            print("Zoom does not match")
        if not registration_ctds[0].shape == registration_ctds[i].shape:
            print("Shape does not match")
        if not registration_ctds[0].orientation == registration_ctds[i].orientation:
            print("Direction does not match")

    #Get the keys that are present in all POIs
    keys = set(registration_ctds[0].keys())
    for ctd in registration_ctds:
        keys = keys.intersection(set(ctd.keys()))
    keys = list(keys)

    ctd = {}
    for key in keys:
        #
        ctd[key] = tuple(np.array([reg_ctd[key] for reg_ctd in registration_ctds]).mean(axis = 0))

    #Sort the new ctd by keys
    ctd = dict(sorted(ctd.items()))
    new_poi = POI(centroids = ctd, orientation = registration_ctds[0].orientation, zoom = registration_ctds[0].zoom, shape = registration_ctds[0].shape)

    return new_poi

def get_ct(container) -> NII:
    ct_query = container.new_query(flatten=True)
    ct_query.filter_format('ct')
    ct_query.filter_filetype('nii.gz') # only nifti files
    ct_candidate = ct_query.candidates[0]

    ct = ct_candidate.open_nii()
    return ct

def get_subreg(container) -> NII:
    subreg_query = container.new_query(flatten=True)
    subreg_query.filter_format('msk')
    subreg_query.filter_filetype('nii.gz') # only nifti files
    subreg_query.filter('seg', 'subreg')
    subreg_candidate = subreg_query.candidates[0]
    
    subreg = subreg_candidate.open_nii()
    return subreg

def get_vertseg(container) -> NII:
    vertseg_query = container.new_query(flatten=True)
    vertseg_query.filter_format('msk')
    vertseg_query.filter_filetype('nii.gz') # only nifti files
    vertseg_query.filter('seg', 'vert')
    vertseg_candidate = vertseg_query.candidates[0]

    vertseg = vertseg_candidate.open_nii()
    return vertseg

def get_files(container, get_poi: callable, get_ct: callable, get_subreg: callable, get_vertseg: callable) -> tuple[POI, NII, NII, NII]:
    return get_poi(container), get_ct(container), get_subreg(container), get_vertseg(container)

def get_bounding_box(mask, vert, margin=5):
        """
        Get the bounding box of a given vertebra in a mask.

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

def process_container(subject, container, save_path: PathLike, rescale_zoom: tuple | None, get_files: Callable[[Subject_Container], tuple[POI, NII, NII, NII]]):
    poi, ct, subreg, vertseg = get_files(container)
    ct.reorient_(('L', 'A', 'S'))
    subreg.reorient_(('L', 'A', 'S'))
    vertseg.reorient_(('L', 'A', 'S'))
    poi.reorient_centroids_to_(ct)
    
    vertebrae = set([key[0] for key in poi.keys()])
    vertseg_arr = vertseg.get_array()

    summary = []
    for vert in vertebrae:
        if vert in vertseg_arr:
            x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box(vertseg_arr, vert)
            ct_path = os.path.join(save_path, subject, str(vert), 'ct.nii.gz')
            subreg_path = os.path.join(save_path, subject, str(vert), 'subreg.nii.gz')
            vertseg_path = os.path.join(save_path, subject, str(vert), 'vertseg.nii.gz')
            poi_path = os.path.join(save_path, subject, str(vert), 'poi.json')

            if not os.path.exists(os.path.join(save_path, subject, str(vert))):
                os.makedirs(os.path.join(save_path, subject, str(vert)))

            ct_cropped = ct.apply_crop_slice(ex_slice = (slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)))
            subreg_cropped = subreg.apply_crop_slice(ex_slice = (slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)))
            vertseg_cropped = vertseg.apply_crop_slice(ex_slice = (slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)))
            poi_cropped = poi.crop_centroids(o_shift = (slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)))

            if rescale_zoom:
                ct_cropped.rescale_(rescale_zoom)
                subreg_cropped.rescale_(rescale_zoom)
                vertseg_cropped.rescale_(rescale_zoom)
                poi_cropped.rescale_(rescale_zoom)

            ct_cropped.save(ct_path, verbose = False)
            subreg_cropped.save(subreg_path, verbose = False)
            vertseg_cropped.save(vertseg_path, verbose = False)
            poi_cropped.save(poi_path, verbose = False)

            #Save the slice indices as json to reconstruct the original POI file (there probably is a more BIDS-like approach to this)
            slice_indices = {
                'x_min': int(x_min),
                'x_max': int(x_max),
                'y_min': int(y_min), 
                'y_max': int(y_max),
                'z_min': int(z_min),
                'z_max': int(z_max)
            }
            with open(os.path.join(save_path, subject, str(vert), 'cutout_slice_indices.json'), 'w') as f:
                json.dump(slice_indices, f)

            summary.append({   
                    'subject': subject,
                    'vertebra': vert,
                    'file_dir': os.path.join(save_path, subject, str(vert)),
            })
        
        else:
            print(f'Vertebra {vert} has no segmentation for subject {subject}')
    
    return summary

def prepare_data(bids_surgery_info: BIDS_Global_info, save_path: str, get_files: callable, rescale_zoom: tuple | None = None, n_workers: int = 8):
        master = []
        partial_process_container = partial(process_container, save_path = save_path, rescale_zoom = rescale_zoom, get_files = get_files)
        master = pqdm(bids_surgery_info.enumerate_subjects(), partial_process_container, n_jobs = n_workers, argument_type = 'args', exception_behaviour='immediate')
        master = [item for sublist in master for item in sublist]
        master_df = pd.DataFrame(master)
        master_df.to_csv(os.path.join(save_path, 'master_df.csv'), index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #Get dataset type (must be gruber or implants)
    parser.add_argument("--dataset_type", type=str, help="The dataset to prepare", choices=["Gruber", "Implants"], required=True)
    parser.add_argument("--data_path", type=str, help="The path to the BIDS dataset", required=True)
    parser.add_argument("--derivatives_name", type=str, help="The name of the derivatives folder", required=True)
    parser.add_argument("--save_path", type=str, help="The path to save the prepared data", required=True)
    parser.add_argument("--no_rescale", action="store_true", help="Whether to skip rescaling the data to isotropic voxels")
    parser.add_argument("--n_workers", type=int, help="The number of workers to use for parallel processing", default=8)
    
    args = parser.parse_args()
    print(args.derivatives_name)

    bids_gloabl_info = BIDS_Global_info(
        datasets = [args.data_path],
        parents = ['rawdata', args.derivatives_name]
    )

    if args.dataset_type == "Gruber":
        get_data_files = partial(get_files, get_poi = get_gruber_poi, get_ct = get_ct, get_subreg = get_subreg, get_vertseg = get_vertseg)
    
    elif args.dataset_type == "Implants":
        get_data_files = partial(get_files, get_poi = get_implants_poi, get_ct = get_ct, get_subreg = get_subreg, get_vertseg = get_vertseg)

    prepare_data(
        bids_surgery_info = bids_gloabl_info,
        save_path = args.save_path,
        get_files = get_data_files,
        rescale_zoom = None if args.no_rescale else (1, 1, 1),
        n_workers = args.n_workers
    )