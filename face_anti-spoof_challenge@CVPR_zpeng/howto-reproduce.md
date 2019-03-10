# How to reproduce on Linux

If you have any questions, please contact us.

--Peng Zhang: m201772781@hust.edu.cn( or peng3.zhang@intel.com)

--Zhiwen Wu: zhiwen.wu@intel.com

--Juan Zhao: juan.j.zhao@intel.com

**Notice: We assume you are in the directory $PROJECT_TOP/face_anti-spoof_challenge@CVPR_zpeng**

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

Download MMFD (链接: https://pan.baidu.com/s/1wCrI7bp3E8-XCctDbGxEVQ 提取码: tqkt OTC-MMFD-11846496, decryption key: OTC-MMFD-11846496)





Uncompressed and copy them to the ./data directory. You can see the contents like below:

```
data
├── fileList.ipynb
├── fileList.py
├── our_filelist
├── our_realsense  # MMFD dataset
├── Testing    #Test set
├── Training    #Train set
├── train_list.txt
├── Val        #Val set
├── val_label.txt #val set with label
└── val_public_list.txt
```

**notice**: Please create a test_private_list.txt file that contains the labels for test dateset, and put it in the ./data directory, we will use it in test phase.
The format of test_private_list.txt file is the same as train_list.txt, like below:

```
Val/0000/000000-color.jpg Val/0000/000000-depth.jpg Val/0000/000000-ir.jpg 0
Val/0000/000001-color.jpg Val/0000/000001-depth.jpg Val/0000/000001-ir.jpg 0
Val/0000/000002-color.jpg Val/0000/000002-depth.jpg Val/0000/000002-ir.jpg 0
Val/0000/000003-color.jpg Val/0000/000003-depth.jpg Val/0000/000003-ir.jpg 0
Val/0000/000004-color.jpg Val/0000/000004-depth.jpg Val/0000/000004-ir.jpg 0
```

And run these commands:
```
cd data
python fileList.py
```

If everything works well, you can see the contents of data directory like these:
```
data
├── 2depth_train.txt
├── 2ir_train.txt
├── 2label_train.txt
├── 2rgb_train.txt
├── depth_test.txt
├── depth_val.txt
├── fileList.ipynb
├── fileList.py
├── ir_test.txt
├── ir_val.txt
├── label_test.txt
├── label_val.txt
├── our_filelist
├── our_realsense  # MMFD dataset
├── rgb_test.txt
├── rgb_val.txt
├── Testing    #Test set
├── test_private_list.txt  # test set list with label
├── Training    #Train set
├── train_list.txt
├── Val        #Val set
├── val_label.txt
└── val_public_list.txt

```

## Train

### Download pretrained models(ImageNet)
download [fishnet150](https://pan.baidu.com/s/1uOEFsBHIdqpDLrbfCZJGUg) pretrained model from [FishNet150 repo](https://github.com/kevin-ssy/FishNet)(Model trained without tricks )

download [mobilenetv2](https://drive.google.com/open?id=1jlto6HRVD3ipNkAl1lNhDbkBp7HylaqR) pretrained model from [MobileNet V2 repo](https://github.com/tonylins/pytorch-mobilenet-v2)

**move them to checkpoints/pre-trainedModels/**

If you want check our pre-trained models,you can download here.(链接: https://pan.baidu.com/s/1eFwtvw4bsLdv5zxtc2tRqQ 提取码: y95k OTC-MMFD-11846496,decryption key:OTC-MMFD-11846496 ) Then move to ./checkpoints directory.

**if you have Multiple gpus, you can use "--gpus num" to train your model in differet gpus.**

### 1. Train FishNet150
```
nohup python main.py --config="cfgs/fishnet150-32.yaml" --b 32 --lr 0.01 --every-decay 30 --fl-gamma 2 >> fishnet150-train.log &
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

### 7.Train MobileLiteNetB using IR image 
```
Step1: Before running train command, please in read_data.py, comment line 11, 12, 21, 22 and uncomment line 15, 16, 25, 26 . 
You code should like below:

# CASIA-SURF training dataset and our private dataset
depth_dir_train_file = os.getcwd() +'/data/2depth_train.txt'
label_dir_train_file = os.getcwd() + '/data/2label_train.txt'

# for IR train
# depth_dir_train_file = os.getcwd() +'/data/ir_final_train.txt'
# label_dir_train_file = os.getcwd() +'/data/label_ir_train.txt'



# CASIA-SURF Val data 
depth_dir_val_file = os.getcwd() +'/data/depth_val.txt'
label_dir_val_file = os.getcwd() +'/data/label_val.txt' #val-label 100%


# depth_dir_val_file = os.getcwd() +'/data/ir_val.txt'
# label_dir_val_file = os.getcwd() +'/data/label_val.txt' #val-label 100%

# # CASIA-SURF Test data 
depth_dir_test_file = os.getcwd() +'/data/depth_test.txt'
label_dir_test_file = os.getcwd() +'/data/label_test.txt'


# depth_dir_test_file = os.getcwd() +'/data/ir_test.txt'
# label_dir_test_file = os.getcwd() +'/data/label_test.txt'

Step2: Running command line. Please note that, use the model "./checkpoints/mobilelitenetB_bs32/_47_best.pth.tar" as the pre-trained model.

nohup python main.py --config="cfgs/MobileLiteNetB-32-ir.yaml" --resume ./checkpoints/mobilelitenetB_bs32/_47_best.pth.tar --lr 0.01 -b 32 --every-decay 100 --fl-gamma 2 >> ir_final_train_mb.log &
```

## Validation


### Predict the validate dataset with several models
we choose these checkpoints to ensemble.And Their performance in validation dataset is as follows.
|model name | ACER|TPR@FPR=10E-2|TPR@FPR=10E-3|FP|FN|epoch|params|FLOPs|
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|FishNet150| 0.00144|0.999668|0.998330|19|0|27|24.96M|6452.72M|
|FishNet150| 0.00181|1.0|0.9996|24|0|52|24.96M|6452.72M|
|FishNet150| 0.00496|0.998664|0.990648|48|8|16|24.96M|6452.72M|
|MobileNet v2|0.00228|0.9996|0.9993|28|1|5|2.23M|306.17M
|MobileNet v2|0.00387|0.999433|0.997662|49|1|6|2.23M|306.17M
|MobileNet v2|0.00402|0.9996|0.992623|51|1|7|2.23M|306.17M
|FeatherNet54|0.00242|1.0|0.99846|32|0|41|0.57M|270.91M|
|FeatherNet54-se|0.00242|1.0|0.996994|32|0|69|0.57M|270.91M|
|MobileLiteNetA|0.00261|1.00|0.961590|19|7|51|0.35M|79.99M|
|MobileLiteNetB|0.00168|1.0|0.997662|20|1|48|0.35M|83.05M|
|**Ensembled all**|0.0000|1.0|1.0|0|0|-|-|-|

you need choose your own checkpoints to resume

### How to create a submission file for validation dataset
```
python main.py --config="cfgs/mobilenetv2.yaml" --resume ./checkpoints/mobilenetv2_bs32/_4_best.pth.tar --val True --val-save True
```

## Test

### Use EnsembleCode_test.ipynb to run test commands and create final submission file
By running the **EnsembleCode_test.ipynb**, the performance results of test set are store in the submission/ directory.
We provided our result of test commands for 10 models predicting test dataset. See test_status.txt

### How to create a submission file for test set
```
python main.py --config="cfgs/mobilenetv2.yaml" --resume ./checkpoints/mobilenetv2_bs32/_4_best.pth.tar --phase-test True --val True --val-save True
```


### How to choose suitable checkpoints to ensemble 
[**important**]

We saved each epoch weight for each model as different checkpoints.
For each model, we select the best checkpoint with the best performance(ACER value) on the validation set, then we choose the secondory checkpoints that are complementary to the best checkpoint. 
The select method of the secondory checkpoints is: 
step-1. select a group checkpoints of good performance(top ACER value, each batch ACC value higher than 90); 
step-2. select out the batches with lower performance of the best checkpoint ; 
step-3. in the selected batches from step two, select out the checkpoints(1 or 2) with the best ACC value. We can treat step-3 as viewing different series(formed by ACC values of those batches the-best-checkpoint didn't perform well), calcualte the max mean value and min standard deviation.
In this way, the selected secondary-checkpoints can be able to provide complementary capabilities to the-best-checkpoint in the batches the-best-checkpoint can not provide good performance.
The performance of the checkpoints we selected on the validation set can be reviewed in the logs/ensemble_model_val_log.md file.


**notice**:You need to replace the path of --resume for your own checkpoints
```
python main.py --config="cfgs/fishnet150-32.yaml" --resume ./checkpoints/fishnet150_bs32/_15.pth.tar --phase-test True --val True --val-save True
python main.py --config="cfgs/fishnet150-32.yaml" --resume ./checkpoints/fishnet150_bs32/_51_best.pth.tar --phase-test True --val True --val-save True
python main.py --config="cfgs/fishnet150-32.yaml" --resume ./checkpoints/fishnet150_bs32/_26_best.pth.tar --phase-test True --val True --val-save True
python main.py --config="cfgs/FeatherNet54-32.yaml" --resume ./checkpoints/FeatherNet54/_40_best.pth.tar --phase-test True --val True --val-save True
python main.py --config="cfgs/FeatherNet54-se-64.yaml" --resume ./checkpoints/FeatherNet54-se/_68_best.pth.tar --phase-test True --val True --val-save True
python main.py --config="cfgs/mobilenetv2.yaml" --resume ./checkpoints/mobilenetv2_bs32/_4_best.pth.tar --phase-test True --val True --val-save True
python main.py --config="cfgs/mobilenetv2.yaml" --resume ./checkpoints/mobilenetv2_bs32/_5.pth.tar --phase-test True --val True --val-save True
python main.py --config="cfgs/mobilenetv2.yaml" --resume ./checkpoints/mobilenetv2_bs32/_6.pth.tar --phase-test True --val True --val-save True
python main.py --config="cfgs/MobileLiteNetA-32.yaml" --resume ./checkpoints/mobilelitenetA_bs32/_50_best.pth.tar --phase-test True --val True --val-save True
python main.py --config="cfgs/MobileLiteNetB-32.yaml" --resume ./checkpoints/mobilelitenetB_bs32/_47_best.pth.tar --phase-test True --val True --val-save True
```

