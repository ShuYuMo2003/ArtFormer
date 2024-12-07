import time
import wandb
import torch
import numpy as np
import lightning as L

from .mylogging import Log, Console

class TransArticulatedBaseDataModule(L.LightningDataModule):
    def __init__(self, d_configs):
        super().__init__()
        self.d_configs = d_configs

class TransArticulatedBaseModule(L.LightningModule):

    def __init__(self, configs):
        super().__init__()
        self.save_hyperparameters()
        self.configs = configs

        # self.wandb_instance = None
        # wandb_config = configs['wandb']
        # if wandb_config['use']:
        #     self.wandb_instance = wandb.init(config=configs, project=wandb_config['project'],
        #                                      entity=wandb_config['entity'], name=time.strftime("%m/%d %I%p:%M:%S"))

        # self.logged_iter_num = 0

    # def log_data(self, data, **kw_args):
    #     if self.wandb_instance is not None:
    #         self.wandb_instance.log(data)

    #     self.log_dict(data, **kw_args)

    #     self.logged_iter_num += 1
    #     text = "Logging step: %s\n" % (self.logged_iter_num)
    #     for k, v in data.items():
    #         _v = v
    #         if isinstance(v, (torch.Tensor)):
    #             _v = v.cpu().detach().numpy()
    #         if isinstance(v, (int, float, str, list)):
    #             text += ("%s: %s\n" % (str(k), str(_v)))
    #     Log.info(text)