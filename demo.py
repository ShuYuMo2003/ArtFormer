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

    multiprocessing.set_start_method("spawn")

    while True:
        tt = time.strftime("%m-%d-%I%p-%M-%S")
        output_path = Path('elog') / f"final_output" / tt
        text_content = input("Input the text prompts:")

        for rep in range(5):
            evaluator.inference_to_output_path(text_content, output_path / str(rep), blender_generated_gif=True)
