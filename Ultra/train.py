import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import TrainDataset
from net.model import UNet
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class UnetModel(pl.LightningModule):
    def __init__(self):
        super(UnetModel,self).__init__()
        self.net = UNet(n_channels=3)
        self.loss_fn  = nn.L1Loss()
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.net(x)
    def training_step(self, batch, batch_idx:int):
        # training_step defines the train loop.
        # it is independent of forward
        (degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored,clean_patch)
        self.log(name="train_loss", value=loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,*args, **kwargs):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)
        return [optimizer],[scheduler]

def main():
    print("Options")
    print(opt)

    trainset = TrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    model = UNet(n_channels=3)
    trainer = pl.Trainer(max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()



