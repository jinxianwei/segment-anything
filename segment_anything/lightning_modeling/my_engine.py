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
                        devices=1)
    trainer.fit(model, datamodule=data_module)
    
    
if __name__ == "__main__":
    model = Segmentation_2d()
    data_module = Segmentation_2D_Datamodule()
    train(model, 20, data_module)