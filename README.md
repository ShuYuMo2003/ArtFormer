# ArtFormer: Controllable Generation of Diverse 3D Articulated Objects

## Clone the repo
```
git clone https://github.com/ShuYuMo2003/ArtFormer.git
```

## Set up Environment

### Conda Env
```
conda env create -f env.yml
conda activate gao
```

Get into `utils/z_to_mesh/utils/libmcubes`, run `python setup.py build_ext --inplace`.
compile `utils/z_to_mesh/utils/libmise` and ``utils/z_to_mesh/utils/libsimplify` with the same command.

### Set up Wandb

We use `wandb` to recording the logs durning the training. You may need to login first.
```
wandb login
```

### Download blender
```
mkdir 3rd && cd 3rd
wget https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.2/blender-4.2.2-linux-x64.tar.xz
tar -xvf blender-4.2.2-linux-x64.tar.xz
```

## Prepare Dataset & Training Model

### Download partnet-mobility-v0.zip
Download the dataset(`partnet-mobility-v0.zip`) from `https://sapien.ucsd.edu/downloads`.
```
cd data/datasets
# place `partnet-mobility-v0.zip` here
unzip partnet-mobility-v0.zip
mv dataset 0_raw_dataset
```

### Prepare SDF Model Dataset & Training

```
cd ../process_data_script
python 1_extract_from_raw_dataset.py
```
The script above will refactor the structure of data in the raw dataset. It's fine if you encounter a little `Failed shape` after the execution, which is caused by some broken objects in PartNet-Mobility.


```
python 2.1_generate_gensdf_dataset.py --n_process 20
```
The script above will sample the $\text{Point Cloud}$ and the $(\text{Position} \in \mathbb{R}^3, \text{SDF} \in \mathbb{R})$ pair for the meshes of each sub-parts of each articulated objectes in the dataset. The script runs on CPU.

This process may take about $30$ mins with `--n_process 20`. If your CPU or memory is insufficient, please reduce `--n_process` appropriately.

After the processing of each mesh, the script will print a table about some related info like:
```
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃     Item      ┃    Shape    ┃ Occ Rate ┃      Bounds      ┃  Abs Sdf Range  ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ Point Uniform │ (200000, 3) │  0.0249  │ -1.0250 ~ 1.0250 │ 0.0000 ~ 1.0486 │
│ Point Surface │ (200000, 3) │  0.4907  │ -1.0470 ~ 1.0485 │ 0.0000 ~ 0.0521 │
│ Point On Mesh │ (144018, 3) │  0.1795  │ -1.0057 ~ 1.0058 │ 0.0000 ~ 0.0000 │
│     Total     │ (544018, 3) │  0.2371  │ -1.0470 ~ 1.0485 │ 0.0000 ~ 1.0486 │
└───────────────┴─────────────┴──────────┴──────────────────┴─────────────────┘
```
If you do see this table, it means everything is working fine. If you never see this table in stdout, it might indicate a **sampling failure!** Please reduce your `--n_process` value appropriately.


Following script will start the training process for SDF model.
```
cd ../.. # back to repo root folder.
python 1_train_SDF.py -c configs/1_SDF/train.yaml
```

### Prepare Diffusion Dataset & Training
After training of SDF model, you can choose a checkpoint of SDF model in `train_root_dir`. The path of checkpoint should be like `train_root_dir/SDF/checkpoint/05-16-02PM-26-34/sdf_epoch=2539-loss=0.00233.ckpt`.

```
cd data/process_data_script
python 2.2_generate_diff_dataset.py --sdf_ckpt_path path/to/SDF/checkpoint
```
The script will generate the dataset for diffusion model.

```
cd ../.. # back to repo root folder.
python 2_train_diff.py -c configs/2_Diff/train.yaml
```

### Prepare Articulation Transformer Dataset & Training

```
cd data/process_data_script
python 3.0_generate_text_used_image.py
```
The script above will generate the image of each object by call `blender` and save the image and log files in `4_screenshot_high_q`.
The images will be used to generate the description of each articulated object.


You can use the script below to call ChatGPT (via. `poe.com`) to generate the description by your self:
```
python 3.1_generate_text_condition.py
```
or use our description dataset:
```
cd ../datasets # you should be in `data/datasets` now
cp ../../attachment/3_text_condition.zip .
unzip 3_text_condition
```

To encode the text description into tokens (by T5 Encoder):
```
python 3.2_generate_encoded_text_condition.py
```

To refactor the dataset into final articulation transformer dataset:
```
python 5_generate_text_transformer_dataset.py --diff_ckpt_path    \
    path/to/diffusion/checkpoint # in train_root_dir/Diff/checkpoint/<datetime>/*, you should choose one.
```

Start training of Articulation Transformer:
```
python 3_train_trans.py -c configs/3_TF-Diff/text-train.yaml
```

## Evaluation
Choose a Articulation Transformer checkpoint in `train_root_dir/Transformer_Diffusion/checkpoint/<training_datetime>/<name>.ckpt` and fill the path of checkpoint into the first line of  `configs/3_TF-Diff/text-eval.yaml`.

Generate the example articulated object by:
```
python 3_pred_trans.py -c configs/3_TF-Diff/text-eval.yaml
```

## Train
1. Training of SDF:
    ```
    python 1_train_SDF.py -c configs/1_SDF/train.yaml
    ```

2. Training of Diffusion:
    ```
    python 2_train_diff.py -c configs/2_Diff/train.yaml
    ```

3. Training of Articulation Transformer:
    For text condition:
    ```
    python 3_train_trans.py -c configs/3_TF-Diff/text-train.yaml
    ```

For image condition:
```
python 3_train_trans.py -c configs/3_TF-Diff/image-train.yaml
```

## Evaluation

```
python 3_pred_trans.py -c configs/3_TF-Diff/<THE-ITEM-YOU-WANT-TO-EVALUATION>.yaml
```
