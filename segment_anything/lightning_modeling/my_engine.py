import os
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.core.datamodule import LightningDataModule
from my_model import Segmentation_2d
from my_datamodule import Segmentation_2D_Datamodule
from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger('./tb_logs', name='segment-anything')

def all_callback_list():
    callback_list = []
    save_best_callback = ModelCheckpoint(monitor='test_dice', mode=
        'max', dirpath='./best_ckpt/', filename=
        '{epoch}_best_{test_dice}', every_n_epochs=1)
    callback_list.append(save_best_callback)
    return callback_list

def train(model, max_epoch, data_module):
    callback_list = all_callback_list()
    trainer = pl.Trainer(logger=logger,
                         max_epochs=max_epoch,
                         accelerator='gpu',
                         devices=1,
                         callbacks=callback_list,
                         log_every_n_steps=1)
    trainer.fit(model, datamodule=data_module)
    
    
if __name__ == "__main__":
    model = Segmentation_2d(lr=2e-4,
                            model_type='vit_b',
                            checkpoint_path='/home/bennie/bennie/temp/segment-anything/sam_vit_b_01ec64.pth')
    data_module = Segmentation_2D_Datamodule(mask_path='/home/bennie/bennie/bennie_project/segment-anything/ground-truth-pixel/',
                                             img_path='/home/bennie/bennie/bennie_project/segment-anything/scans/',
                                             split_test=0.2,
                                             batch_size=8,
                                             num_workers=16,
                                             random_seed=2023)
    train(model, 20, data_module)