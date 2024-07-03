import pytorch_lightning as pl
import torch
import numpy as np

import modules.FeatureExtractionModules as feat_modules
import modules.RefinementModules as ref_modules

class PoiPredictionModule(pl.LightningModule):
    def __init__(
            self, 
            coarse_config, 
            refinement_config, 
            lr=1e-4, 
            loss_weights=[1, 1],
            optimizer='AdamW',
            scheduler_config=None,
            feature_freeze_patience=None
            ):
        super().__init__()
        self.feature_extraction_module = create_feature_extraction_module(coarse_config)
        self.refinement_module = create_refinement_module(refinement_config)
        self.lr = lr
        self.loss_weights = torch.tensor(loss_weights) / torch.sum(torch.tensor(loss_weights))
        self.feature_freeze_patience = feature_freeze_patience
        self.best_feature_loss = np.inf
        self.val_feature_loss_outputs = []
        self.epochs_without_improvement = 0
        self.feature_extactor_frozen = False
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config

        #Save hyperparameters
        self.save_hyperparameters()

    def forward(self, batch):
        batch = self.feature_extraction_module(batch)
        batch = self.refinement_module(batch)

        return batch

    
    def training_step(self, batch):
        batch = self(batch)
        #Calculate the feature extraction loss
        feature_loss = self.feature_extraction_module.calculate_loss(batch)
        #Calculate the refinement loss
        refinement_loss = self.refinement_module.calculate_loss(batch)
        loss = feature_loss * self.loss_weights[0] + refinement_loss * self.loss_weights[1]

        metrics = self.calculate_metrics(batch, 'train')
        batch_size = batch['input'].shape[0]

        self.log('train_loss', loss, on_epoch=True, batch_size=batch_size)
        self.log_dict(metrics, on_epoch=True, on_step=False, batch_size=batch_size)

        return loss
    
    def validation_step(self, batch):
        batch = self(batch)
        #Calculate the feature extraction loss
        feature_loss = self.feature_extraction_module.calculate_loss(batch)
        #Calculate the refinement loss
        refinement_loss = self.refinement_module.calculate_loss(batch)
        loss = feature_loss * self.loss_weights[0] + refinement_loss * self.loss_weights[1]

        metrics = self.calculate_metrics(batch, 'val')
        batch_size = batch['input'].shape[0]

        self.val_feature_loss_outputs.append(feature_loss)

        self.log('val_feature_loss', feature_loss, on_epoch=True, batch_size=batch_size)
        self.log('val_refinement_loss', refinement_loss, on_epoch=True, batch_size=batch_size)
        self.log('val_loss', loss, on_epoch=True, batch_size=batch_size)
        self.log_dict(metrics, on_epoch=True, on_step=False, batch_size=batch_size)

        return loss
    
    def on_validation_epoch_end(self):
        #Check if the feature extraction module should be frozen
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
                if self.epochs_without_improvement >= self.feature_freeze_patience and not self.feature_extactor_frozen:
                    self.freeze_feature_extractor()
                    self.feature_extactor_frozen = True
                    print("Feature extraction module frozen")


    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.optimizer)
        optimizer = optimizer_class(self.parameters(), lr=self.lr)

        if self.scheduler_config:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.scheduler_config['type'])
            scheduler = scheduler_class(optimizer, **self.scheduler_config['params'])
            
            scheduler_config = {'scheduler': scheduler, 'interval': 'epoch'}
            if 'monitor' in self.scheduler_config:
                scheduler_config['monitor'] = self.scheduler_config['monitor']

            return [optimizer], [scheduler_config]
        
        return optimizer
    
    def calculate_metrics(self, batch, mode):
        feature_metrics = self.feature_extraction_module.calculate_metrics(batch, mode)
        refinement_metrics = self.refinement_module.calculate_metrics(batch, mode)

        return {**feature_metrics, **refinement_metrics}
    
    def freeze_feature_extractor(self):
        self.log('feature_frozen', True, on_epoch=True)
        for param in self.feature_extraction_module.parameters():
            param.requires_grad = False


def create_feature_extraction_module(config):
    module_type = getattr(feat_modules, config['type'])
    if module_type is None:
        raise ValueError(f"Unknown feature extraction module type: {config['type']}")
    
    return module_type(**config['params'])

def create_refinement_module(config):
    module_type = getattr(ref_modules, config['type'])
    if module_type is None:
        raise ValueError(f"Unknown refinement module type: {config['type']}")
    
    return module_type(**config['params'])