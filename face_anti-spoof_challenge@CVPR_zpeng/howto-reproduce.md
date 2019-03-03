# How to reproduce on Linux

## Prerequisites

### Install conda
Follow https://conda.io/projects/conda/en/latest/user-guide/install/index.html

###  Create conda environment and install requeirements
```
conda env create -n env_name -f env.yml
```

###  Activate conda environment
```
conda activate env_name
```

### Data preparation
We use 2 dataset for training our models. One is CASIA-SURF, another is created by ourselves, named Multi-Modality Face Dataset', abbreviated as MMFD.

Download CASIA-SURF

Download MMFD(TODO: add url here).

Uncompressed and copy them to the data directory and run these commands (assume you are already in the source directory):
```
cd data
python fileList.py
```

If everything works well, you can see the contents of data directory like these:
```
├── data
│   ├── our_realsense # MMFD dataset
│   ├── Training
│   ├── Val
│   ├── Testing
│   ├── our_filelist
│   ├── fileList.ipynb
│   ├── fileList.py
│   ├── train_list.txt
│   ├── val_label.txt
│   ├── val_public_list.txt 
│   ├── test_private_list.txt
```

## Train

### Download pretrained models
Download [fishnet150](https://pan.baidu.com/s/1uOEFsBHIdqpDLrbfCZJGUg) pretrained model from [FishNet150](https://github.com/kevin-ssy/FishNet)(Model trained without tricks )

Download [mobilenetv2](https://drive.google.com/open?id=1jlto6HRVD3ipNkAl1lNhDbkBp7HylaqR) pretrained model from [MobileNet V2](https://github.com/tonylins/pytorch-mobilenet-v2)

**move them to checkpoints/pre-trainedModels/**

### 1. Train FishNet150
```
nohup python main.py --config="cfgs/fishnet150-32-5train.yaml" --b 32 --lr 0.01 --every-decay 30 --fl-gamma 2 >> fishnet150-train.log &
```

### 2. Train MobileNet V2
```
nohup python main.py --config="cfgs/mobilenetv2.yaml" --b 32 --lr 0.01 --every-decay 40 --fl-gamma 2 >> mobilenetv2-bs32-train.log &
```

### 3. Train FeatherNet54
```
nohup python main.py --config="cfgs/FeatherNet54-32.yaml" --every-decay 60 -b 32 --lr 0.01 --fl-gamma 3 >>FNet54-bs32-train.log &
```

### 4. Train FeatherNet54-SE
```
nohup python main.py --config="cfgs/FeatherNet54-se-64.yaml" --b 64 --lr 0.01  --every-decay 60 --fl-gamma 3 >> FNet54-se-bs64-train.log &
```

### 5. Train MobileLiteNetA
```
nohup python main.py --config="cfgs/MobileLiteNetA-32.yaml" --b 32 --lr 0.01  --every-decay 60 --fl-gamma 3 >> MobileLiteNetA-bs32-train.log &
```

### 6. Train MobileLiteNetB
```
nohup python main.py --config="cfgs/MobileLiteNetB-32.yaml" --b 32 --lr 0.01  --every-decay 60 --fl-gamma 3 >> MobileLiteNetB-bs32--train.log &
```

### How to create a submission file
```
python main.py --config="cfgs/mobilenetv2.yaml" --resume ./checkpoints/mobilenetv2_bs32/_4_best.pth.tar --val True --val-save True
```

## Test
### Predict the test set with several models
By running the following commands, the performance results of test set are store in the submission/ directory.

```
python main.py --config="cfgs/fishnet150-32.yaml" --resume ./checkpoints/fishnet150_bs32/_15_best.pth.tar --phase-test True --val True --val-save True
python main.py --config="cfgs/fishnet150-32.yaml" --resume ./checkpoints/fishnet150_bs32/_51_best.pth.tar --phase-test True --val True --val-save True
python main.py --config="cfgs/fishnet150-32.yaml" --resume ./checkpoints/fishnet150_bs32/_10_best.pth.tar --phase-test True --val True --val-save True
python main.py --config="cfgs/FeatherNet54-32.yaml" --resume ./checkpoints/FeatherNet54/_40_best.pth.tar --phase-test True --val True --val-save True
python main.py --config="cfgs/FeatherNet54-se-64.yaml" --resume ./checkpoints/FeatherNet54-se/_68_best.pth.tar --phase-test True --val True --val-save True
python main.py --config="cfgs/mobilenetv2.yaml" --resume ./checkpoints/mobilenetv2_bs32/_4_best.pth.tar --phase-test True --val True --val-save True
python main.py --config="cfgs/mobilenetv2.yaml" --resume ./checkpoints/mobilenetv2_bs32/_5.pth.tar --phase-test True --val True --val-save True
python main.py --config="cfgs/mobilenetv2.yaml" --resume ./checkpoints/mobilenetv2_bs32/_6.pth.tar --phase-test True --val True --val-save True
python main.py --config="cfgs/MobileLiteNetA-32.yaml" --resume ./checkpoints/mobilelitenetA_bs32/_50_best.pth.tar --phase-test True --val True --val-save True
python main.py --config="cfgs/MobileLiteNetB-32.yaml" --resume ./checkpoints/mobilelitenetB_bs32/_47_best.pth.tar --phase-test True --val True --val-save True
```

### Generate the final submission by assemble results from above models
```
python gen_final_submission.py
```
