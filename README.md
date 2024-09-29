# Automated Point-of-Interest Prediction on CT Scans of Human Vertebrae Using Spine Segmentations

Welcome to the code repository of my master thesis at TUM. This project addresses the challenge of predicting ligament attachment points on vertebrae in CT Scans of the spine. Accurate identification of these points is crucial for understanding spinal anatomy and facilitating patient-specific simulations that may uncover risk factors of low back pain.

For a quick overview of the project, view the [poster](doc/PresentationPosterMA.pdf). For an in-depth exploration, refer to my [thesis](doc/Daniel_Regenbrecht_Master_Thesis_final_signed.pdf).

## Project Description

The project comprises of three major components:

- Data analysis and preparation
- Training a POI prediction model
- An inference pipeline for a given sample/dataset

## Installation (Ubuntu)

Create and activate a virtual environment, e.g. by running

```bash
conda create -n poi-prediction python=3.10
cona activate poi-prediction
```

Set the python interpreter path in your IDE to the version in the environment, i.e. the output of

```bash
which python
```

Install the BIDS toolbox as provided in this repository

```bash
unzip bids.zip
cd bids
python setup.py build
python setup.py install
```

Back in the project directory, install the required packages 

```bash
pip install -r requirements.txt
```

## Training your Own Model

To train your own model, a dataset in BIDS-like format is required, i.e. the following structure is expected:

```text
dataset-folder
├── <rawdata>
    ├── subfolders
        ├── CT image file
├── <derivatives>
    ├── subfolders
        ├── Instance Segmentation Mask
        ├── Subregion Segmentaiton Mask
        ├── POI file
```

For each CT file, corresponding segmentation files and a POI file are required.

Since the size of the CT scans and segmentations is generally too large to fit into GPU memory, the model predicts the POIs on a vertebra-level instead of processing an entire spine segmentation at once. Therefore, it is necessary to cut out individual vertebrae from the large image and masks and shift the POI file accordingly, which the instance mask is used for. Further, the images are brought into standard orientation and scale. To avoid repetitive computations during training, these preprocessing steps are carried out in bulk and the cutouts are saved to the disk. In order to run the bulk preprocessing, enter the src folder and run prepare_data.py.

WARNING: By default, this step uses 8 CPU cores in parallel to speed up the pre-processing. You can specify a different number with the --n_workers argument. This step will run for several minutes to hours and may require several GB of disk space depending on the size of the dataset (The cutouts of the Ligament Attachment Point dataset used in the development of this project require 465MB of disk space for 36 initial scans)

```bash
cd src
python3 prepare_data.py --data_path $PATH_TO_YOUR_DS --derivatives_name $NAME_OF_DERIVATIVES_FOLDER --save_path $PATH_TO_SAVE_CUTOUS
```

Along with the cutouts, the script saves a csv file containing the paths of all cutouts and a json file specifying the parameters used for the creation of cutouts to reliably create appropriate cutouts during inference.

Once cutouts are created, you can start training. Create a training config (samples can be found in the experiment_config subdirectory) to specify the location of data, logging, as well as data and model configurations. Then, inside the src folder, run

```bash
train.py --config $PATH_TO_YOUR_CONFIG_FILE
```

You can also run several trainings consecutively by placing the respective config files in one directory and using the --config-dir argument instead of --config in the above call. Further, training can be run using cross-validation by using train_cv.py instead of train.py. In this mode, training and validtation split in the config will be treated the same and random splits will be created for each fold (adjustable using the --n_folds argument)

## Inferring with a Trained Model

tbc

## Example Usage (Inference on VerSe19)

tbc
