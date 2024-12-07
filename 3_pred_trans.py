from utils import parse_config_from_args
from lightning.pytorch import seed_everything
from model.Transformer.eval import Evaluater
from utils.mylogging import Log
from pathlib import Path

from rich import print
from tqdm import tqdm
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


    OPTION = config['OPTION']
    if OPTION == -1:
        print("Running evaluation for result evaluation.")
        output_path = Path('data/datasets') / config['evaluation_output_path']
        shutil.rmtree(output_path, ignore_errors=True)
        output_path.mkdir(exist_ok=True, parents=True)

        from evaluation_need_list import selected_for_eval_file_name
        import numpy as np
        import torch

        for object_text_pair in selected_for_eval_file_name['fn_list']:
            object_text_pair_info = object_text_pair.split('_')
            shape_name, text_idx = '_'.join(object_text_pair_info[:2]), object_text_pair_info[2]
            encoded_text_path = Path('data/datasets/3_encoded_text_condition') / (shape_name + '_' + text_idx + ".npy")
            encoded_text = np.load(encoded_text_path, allow_pickle=True).item()['encoded_text']
            encoded_text = torch.tensor(encoded_text).type(torch.float32)

            cur_output_path = output_path / ((shape_name + '_' + text_idx) + '.dat')
            evaluator.inference_dat_file_only('', cur_output_path, encoded_text)

    elif OPTION == 0:
        for obj_path in tqdm(obj_paths, 'obj count'):
            obj_name = obj_path.stem
            for text_path in tqdm(list(obj_path.glob('*')), 'text count'):
                text_name = text_path.stem
                input_text = text_path.read_text()
                output_path = Path('elog') / tt / f"{obj_name}_{text_name}"
                Log.info("Processing opt=%s ipt=%s", output_path, text_path)
                evaluator.inference_to_output_path(input_text, output_path)
    elif OPTION == 1:
        obj_name_list = ['StorageFurniture_45243_1',
                         'StorageFurniture_45374_2', 'StorageFurniture_45636_3',
                         'StorageFurniture_45710_0', 'StorageFurniture_46466_2',
                         'StorageFurniture_46856_0', 'StorageFurniture_46856_1']

        for obj_name in tqdm(obj_name_list, 'obj_list'):
            output_path = Path('elog') / f"Final_OP1_{tt}" / f"{obj_name}"
            obj_infos = obj_name.split('_')
            text_content = (text_datasets / '_'.join(obj_infos[:2]) / (str(obj_infos[2])+'.txt')).read_text()
            print("Processing", obj_name)
            for rep in range(20):
                evaluator.inference_to_output_path(text_content, output_path / str(rep), blender_generated_gif=True)
    elif OPTION == 2:

        for obj_name in ['StorageFurniture_45636_3', 'Bottle_3571_1']:
            output_path = Path('elog') / f"Final_OP1_SF_45636_3_{tt}" / f"{obj_name}"
            obj_infos = obj_name.split('_')
            text_content = (text_datasets / '_'.join(obj_infos[:2]) / (str(obj_infos[2])+'.txt')).read_text()

            for rep in range(100):
                atten_weights_list = evaluator.inference_to_output_path(text_content, output_path / str(rep), blender_generated_gif=True)

                import pickle
                with open(output_path / str(rep) / 'atten_data.pkl', 'wb') as file:
                    pickle.dump(atten_weights_list, file)
    elif OPTION == 3: # evaluation image condition
        encoded_image_path_list = list(Path('data/datasets/5_screenshot_encoded_real').glob('*.npy'))

        import random
        random.shuffle(encoded_image_path_list)

        output_path = Path('e_image_output') / tt
        output_path.mkdir(exist_ok=True, parents=True)

        for encoded_image_path in tqdm(encoded_image_path_list, "Image Condition"):
            name = encoded_image_path.stem
            encoded_image = np.load(encoded_image_path, allow_pickle=True)
            encoded_image = torch.tensor(encoded_image).type(torch.float32)
            evaluator.inference_to_output_path('', output_path / name , enc_data=encoded_image, blender_generated_gif=True)
    elif OPTION == 4: # real image conditions.
        encoded_image_path_list = list(Path('data/datasets/6_encoded_real_real_image').glob('*.npy'))

        import random
        random.shuffle(encoded_image_path_list)

        output_path = Path('e_image_output') / tt
        output_path.mkdir(exist_ok=True, parents=True)

        for encoded_image_path in tqdm(encoded_image_path_list, "Image Condition"):
            name = encoded_image_path.stem
            encoded_image = np.load(encoded_image_path, allow_pickle=True)
            encoded_image = torch.tensor(encoded_image).type(torch.float32)
            evaluator.inference_to_output_path('', output_path / name , enc_data=encoded_image, blender_generated_gif=True)

