import os
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.core.datamodule import LightningDataModule
from my_model import Segmentation_2d
from my_datamodule import Segmentation_2D_Datamodule

def train(model, max_epoch, data_module):
    trainer = pl.Trainer(max_epochs=max_epoch,
                        accelerator='gpu',
                        devices=2)
    trainer.fit(model, datamodule=data_module)
    
    
if __name__ == "__main__":
    model = Segmentation_2d(lr=1e-5,
                            model_type='vit_b',
                            checkpoint_path='/home/bennie/bennie/temp/segment-anything/sam_vit_b_01ec64.pth')
    data_module = Segmentation_2D_Datamodule(mask_path='/home/bennie/bennie/bennie_project/segment-anything/ground-truth-pixel/',
                                             img_path='/home/bennie/bennie/bennie_project/segment-anything/scans/',
                                             split_test=0.2,
                                             batch_size=8,
                                             num_workers=16,
                                             random_seed=2023)
    train(model, 20, data_module)