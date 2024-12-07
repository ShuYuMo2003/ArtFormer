# ArtFormer: Controllable Generation of Diverse 3D Articulated Objects

## Environment
```
conda env create -f env.yml
```

## Data Preprocess
donwload the dataset of [PartNet-Mobility](https://sapien.ucsd.edu/downloads) unzip it in `data/datasets/0_raw_dataset`. Then excute the scripts in `data/process_data_script` one by one.

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
