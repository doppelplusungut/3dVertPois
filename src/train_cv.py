import argparse
import json
import os

import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold

from eval import create_self_training_pois
from modules.PoiDataModules import create_data_module
from modules.PoiModule import PoiPredictionModule


def save_data_module_config(data_module, save_path):
    """Save the hyperparameters of the DataModule to a JSON file for docuementation and
    easy reproducibility."""
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)
    # Save the hyperparameters to a JSON file
    with open(os.path.join(save_path, "data_module_params.json"), "w") as f:
        json.dump(data_module.hparams, f, indent=4)


def create_callbacks(callbacks_config):
    callbacks_list = []
    for callback_config in callbacks_config:
        callback_type = callback_config["type"]
        if callback_type == "ModelCheckpoint":
            callbacks_list.append(
                pl.callbacks.ModelCheckpoint(**callback_config["params"])
            )
        elif callback_type == "EarlyStopping":
            callbacks_list.append(
                pl.callbacks.EarlyStopping(**callback_config["params"])
            )
        # Add other callbacks as needed
    return callbacks_list


def run_cv(n_folds, experiment_config, save_predictions=False, poi_file_ending=None):
    # If the predictions are saved the file ending must be set
    if save_predictions and not poi_file_ending:
        raise ValueError("If predictions are saved the poi file ending must be set")
    # Set the matmul precision to 'medium' for better performance
    torch.set_float32_matmul_precision("medium")

    train_subjects = experiment_config["data_module_config"]["params"]["train_subjects"]
    val_subjects = experiment_config["data_module_config"]["params"]["val_subjects"]

    # Add val subjects to train subjects and create random folds
    train_subjects += val_subjects
    kf = KFold(n_splits=n_folds, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_subjects)):
        train_subjects_fold = [train_subjects[i] for i in train_idx]
        val_subjects_fold = [train_subjects[i] for i in val_idx]

        data_module_config = experiment_config["data_module_config"]
        data_module_config["params"]["train_subjects"] = train_subjects_fold
        data_module_config["params"]["val_subjects"] = val_subjects_fold

        data_module = create_data_module(data_module_config)
        data_module.setup()
        poi_module = PoiPredictionModule(**experiment_config["module_config"]["params"])

        # Create callbacks from configuration
        callbacks = create_callbacks(experiment_config.get("callbacks_config", []))

        # Trainer configuration
        trainer_config = experiment_config.get("trainer_config", {})
        trainer_config["callbacks"] = callbacks

        # Add fold to path
        path = experiment_config["path"] + f"/fold_{fold}"
        trainer_config["logger"] = pl.loggers.TensorBoardLogger(
            path, name=experiment_config["name"]
        )

        trainer = pl.Trainer(**trainer_config)

        # Save DataModule config
        data_module_config_path = trainer.logger.log_dir
        save_data_module_config(data_module, data_module_config_path)

        trainer.fit(
            poi_module,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
        )

        if save_predictions:
            # Get path of best model
            best_model_path = trainer.checkpoint_callback.best_model_path

            create_self_training_pois(
                data_module_save_path=os.path.join(
                    data_module_config_path, "data_module_params.json"
                ),
                checkpoint_path=best_model_path,
                poi_file_ending=poi_file_ending,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_folds", type=int, help="Number of folds for cross-validation"
    )
    parser.add_argument(
        "--save-predictions", action="store_true", help="Save predictions for each fold"
    )
    parser.add_argument(
        "--poi-file-ending",
        type=str,
        help="Ending of the poi file to save predictions to",
    )
    parser.add_argument("--config", type=str, help="Experiment config file")
    parser.add_argument(
        "--config-dir", type=str, help="Directory containing experiment config files"
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            experiment_config = json.load(f)
            run_cv(
                args.n_folds,
                experiment_config,
                save_predictions=args.save_predictions,
                poi_file_ending=args.poi_file_ending,
            )

    if args.config_dir:
        for config_file in os.listdir(args.config_dir):
            with open(os.path.join(args.config_dir, config_file), "r") as f:
                experiment_config = json.load(f)
                run_cv(
                    args.n_folds,
                    experiment_config,
                    save_predictions=args.save_predictions,
                    poi_file_ending=args.poi_file_ending,
                )
