import time
import torch

from torch.utils.data import DataLoader

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, ModelSummary

from utils.mylogging import Log
from utils import parse_config_from_args

from pathlib import Path
from model.Diffusion import Diffusion
from model.Diffusion.dataset import DiffusionDataset

import os
# Set WANDB_CACHE_DIR to a local directory to avoid no space left error
# cmd to clean up the cache: wandb artifact cache cleanup 1GB
os.environ['WANDB_CACHE_DIR'] = (Path() / 'wandb/cache').resolve().as_posix()
os.environ['WANDB_DATA_DIR'] = (Path() / 'wandb/data').resolve().as_posix()
import wandb

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    run_name = time.strftime("%m-%d-%I%p-%M-%S")
    config = parse_config_from_args()
    seed_everything(config['seed'])
    if config['wandb']['use']:
        wandb_run = wandb.init(
            config=config, project=config['wandb']['project'],
            entity=config['wandb']['entity'], name=run_name,

        )
        wandb_logger = WandbLogger()

    # Configure data module
    d_configs = config['dataset_n_dataloader']
    dataloader = DataLoader(DiffusionDataset(dataset_path=Path(d_configs['dataset_path'])),
                        num_workers=d_configs['n_workers'], batch_size=d_configs['batch_size'],
                        drop_last=True, shuffle=True, pin_memory=True, persistent_workers=True)
    config['evaluation']['sdf_model_path'] = Path(dataloader.dataset.get_gensdf_ckpt_path())

    print("Len = ", len(dataloader.dataset))

    # Set up model
    if config.get('load_from_pretrained_ckpt') is not None:
        Log.info("Load pretrained checkpoint %s", config['load_from_pretrained_ckpt'])
        model = Diffusion.load_from_checkpoint(config['load_from_pretrained_ckpt'])
        Log.info('Done')
    else:
        model = Diffusion(config)

    # Configure save checkpoint callback
    checkpoint_callback = ModelCheckpoint(
            save_top_k=-1,
            every_n_epochs=config['checkpoint']['freq'],
            dirpath=config['checkpoint']['path'] + '/' + run_name,
            filename="diffusion-{epoch:04d}-{loss:.5f}",
        )

    # Configure trainer
    optional_kw_args = dict()
    if config['wandb']['use']:
        optional_kw_args['logger'] = wandb_logger

    trainer = Trainer(devices=config['devices'], accelerator=config["accelerator"],
                      benchmark=True,
                      callbacks=[ModelSummary(max_depth=1), checkpoint_callback, TQDMProgressBar()],
                      check_val_every_n_epoch=config['evaluation']['freq_epoch'],
                      default_root_dir=config['default_root_dir'],
                      max_epochs=config['num_epochs'], profiler="simple",
                      log_every_n_steps=10,
                      **optional_kw_args)

    Log.info("Start training...")

    trainer.fit(model=model, train_dataloaders=dataloader, val_dataloaders=dataloader)