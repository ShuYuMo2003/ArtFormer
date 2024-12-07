import time
import torch
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
from rich import print
from pathlib import Path
from transformers import Blip2Processor, Blip2Model

import sys
sys.path.append('../..')
from utils.mylogging import Log, console

blip_cache_path = Path('../cache/blip_cache')
blip_model_name = "Salesforce/blip2-opt-2.7b"

screenshot_output_path = Path('../datasets/6_encoded_real_real_image')
screenshot_input_path = Path('../datasets/6_real_real_image')

def generate_high_q_screenshot_latent_code(image_path: Path, output_path: Path, obj_name: str, blip_processor, blip_model):
    image = Image.open(image_path.as_posix())

    inputs = blip_processor(images=image, return_tensors="pt").to(blip_device, torch.float16)
    output = blip_model.get_image_features(**inputs)
    image_feature = output.last_hidden_state

    image_feature = image_feature.detach().cpu().numpy()

    np.save(output_path / f'{obj_name}.npy', image_feature.squeeze(0))

if __name__ == '__main__':
    blip_cache_path.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(screenshot_output_path, ignore_errors=True)
    screenshot_output_path.mkdir(parents=True, exist_ok=True)

    blip_device = "cuda" if torch.cuda.is_available() else "cpu"

    blip_processor = Blip2Processor.from_pretrained(blip_model_name, cache_dir=blip_cache_path.as_posix())
    blip_model     = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=blip_cache_path.as_posix(), torch_dtype=torch.float16).to(blip_device)
    images_path    = list(screenshot_input_path.glob('*.png'))

    for image_path in tqdm(images_path):
        obj_name = image_path.stem
        start_time = time.time()
        generate_high_q_screenshot_latent_code(image_path, screenshot_output_path,
                                               obj_name, blip_processor, blip_model)
        Log.info(f'[{obj_name}] Done in {time.time() - start_time:.2f}s')