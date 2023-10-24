import random
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split
import torchvision.transforms as transforms

from my_dataset import Segmentation_2D_Dataset
from torch.utils.data import random_split

class Segmentation_2D_Datamodule(pl.LightningDataModule):
    """Pytorch_lightning.LightningDataModule
    """
    def __init__(self,
                 split_test: float = 0.2,
                 batch_size: int = 2,
                 num_workers: int = 0,
                 random_seed: int = 2023):
        super().__init__()
        self.split_test = split_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed

    def setup(self, stage=None):
        torch.manual_seed(2023) # 固定随机种子
        MyDataset = Segmentation_2D_Dataset()
        test_size = int(self.split_test * len(MyDataset))
        train_size = len(MyDataset) - test_size
        if stage == 'fit':
            self.train_dataset, self.test_dataset = random_split(MyDataset, [train_size, test_size])
            
            

    def train_dataloader(self):
        # train_loader = DataLoader(self.train_dataset,
        #                           batch_size=self.batch_size,
        #                           shuffle=True,
        #                           num_workers=self.num_workers)
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True)
        return train_loader

    def val_dataloader(self):
        # test_loader = DataLoader(self.test_dataset,
        #                          batch_size=self.batch_size,
        #                          shuffle=False,
        #                          num_workers=self.num_workers)
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False)
        return test_loader