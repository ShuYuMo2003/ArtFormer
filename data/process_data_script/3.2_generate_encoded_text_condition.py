import torch
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, T5EncoderModel

def process_description(t5_cache_path:Path, t5_model_name:str,
                        batch_size:int, device:str, max_sentence_length:int,
                        description_path:Path, encoded_text_path:Path):

    print('Processing description: ', '\n','description_path: ', description_path, '\n',
          'Using model: ', t5_model_name, '\n', 'Batch size: ', batch_size, '\n', 'Device: ', device)
    tokenizer = AutoTokenizer.from_pretrained(t5_model_name, cache_dir=t5_cache_path.as_posix())
    model = T5EncoderModel.from_pretrained(t5_model_name, cache_dir=t5_cache_path.as_posix()).to(device)

    shapes_description_path = glob((description_path / "*").as_posix())

    shape_to_description = {}
    for shape_desc_path in shapes_description_path:
        all_descriptions_path = glob((Path(shape_desc_path) / "*.txt").as_posix())
        descriptions = []
        for desc_path in all_descriptions_path:
            text = Path(desc_path).read_text()
            assert "try again" not in text and "unable" not in text and "sorry" not in text
            descriptions.append(text)

        shape_name = shape_desc_path.split('/')[-1]
        shape_to_description[shape_name] = descriptions

    pure_sentences = []
    for shape_name, descriptions in shape_to_description.items():
        for idx, desc in enumerate(descriptions):
            pure_sentences.append((encoded_text_path / (shape_name + '_' + str(idx) + '.npy'), desc))

    for s in tqdm(range(0, len(pure_sentences), batch_size), desc="Encoding sentences"):
        slice = pure_sentences[s:min(s+batch_size, len(pure_sentences))]
        save_path = [x[0] for x in slice]
        input = [x[1] for x in slice]

        input_ids = tokenizer(input, return_tensors="pt", padding='max_length',
                              max_length=max_sentence_length, truncation=True).input_ids
        input_ids = input_ids.to(device)
        outputs = model(input_ids=input_ids)
        encoded_text = outputs.last_hidden_state.detach().cpu().numpy()

        for idx, path in enumerate(save_path):
            print('[Write]', path)
            np.save(path, dict(encoded_text=encoded_text[idx, ...], text=input[idx]))

if __name__ == '__main__':
    t5_cache_path = Path('../../cache/t5_cache')
    t5_cache_path.mkdir(exist_ok=True)
    t5_model_name = 'google-t5/t5-large'
    t5_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t5_batch_size = 4
    t5_max_sentence_length = 512

    description_path = Path('../datasets/3_text_condition')
    description_path.mkdir(exist_ok=True)

    encoded_text_path = Path('../datasets/3_encoded_text_condition')
    shutil.rmtree(encoded_text_path, ignore_errors=True)
    encoded_text_path.mkdir(exist_ok=True)


    shapes_encoded_text = process_description(t5_cache_path, t5_model_name,
                                              t5_batch_size, t5_device,
                                              t5_max_sentence_length,
                                              description_path,
                                              encoded_text_path)