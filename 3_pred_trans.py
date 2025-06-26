from utils import parse_config_from_args
from lightning.pytorch import seed_everything
from model.Transformer.eval import Evaluater
from utils.mylogging import Log
from pathlib import Path

from rich import print
from tqdm import tqdm

import multiprocessing
import numpy as np
import time
import random
import shutil
import torch

if __name__ == '__main__':
    config = parse_config_from_args()
    Log.info(f'Loading : {Evaluater}')
    evaluator = Evaluater(config)

    text_datasets = Path('data/datasets/3_text_condition')

    obj_paths = list(text_datasets.glob('*'))
    random.shuffle(obj_paths)
    tt = time.strftime("%m-%d-%I%p-%M-%S")

    multiprocessing.set_start_method("spawn")

    OPTION = config['OPTION']
    if OPTION == 1:
        obj_name_list_all = list(map(lambda x : x.parent.stem + '_' + x.stem,
                                     Path('data/datasets/3_text_condition').glob('*/*')))
        obj_name_list = random.choices(obj_name_list_all, k=100)
        random.shuffle(obj_name_list)

        for obj_name in tqdm(obj_name_list, 'obj_list'):
            output_path = Path('elog') / f"final_output" / f"{obj_name}"
            obj_infos = obj_name.split('_')
            text_content = (text_datasets / '_'.join(obj_infos[:2]) / (str(obj_infos[2])+'.txt')).read_text()
            print("Processing", obj_name)

            for rep in range(7):
                evaluator.inference_to_output_path(text_content, output_path / str(rep), blender_generated_gif=True)
    else:
        print('NOT SUPPORT ANYMORE.')
