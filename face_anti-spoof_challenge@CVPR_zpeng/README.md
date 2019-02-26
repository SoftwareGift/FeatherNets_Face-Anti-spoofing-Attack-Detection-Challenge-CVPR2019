## FeatherNet for [Face Anti-spoofing Attack Detection Challenge@CVPR2019](https://competitions.codalab.org/competitions/20853#results)[1]

# Results on the validation set
|model name | ACER|TPR@FPR=10E-2|TPR@FPR=10E-3|FP|FN|epoch|
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|FishNet150| 0.00181|1.0|0.9996|24|0|52|
|FishNet150| 0.00496467|0.998664|0.990648|48|8|16|
|FishNet150| 0.01023724|0.995992|0.890782|131|2|11|
|MobileNet v2|0.00228|0.9996|0.9993|28|1|5|
|MobileNet v2|0.00387126|0.999433|0.997662|49|1|6|
|MobileNet v2|0.00402246|0.9996|0.992623|51|1|7|
|FeatherNet54|0.002419|1.0|0.99846|32|0|41|
|FeatherNet54-se|0.002419|1.0|0.996994|32|0|69|
|**Ensembled all**|0.0000|1.0|1.0|0|0|-|




# Prerequisites

##  install requeirements
```
pip  install   -r requeirements.txt
```


# Data


### [CASIA-SURF Dataset](https://arxiv.org/abs/1812.00408)[2]


### Our Private Dataset(Available Soon)



### Data index tree
```
├── data
│   ├── our_realsense
│   ├── Training
│   ├── Val
```

### Data Augmentation

| Method | Settings |
| -----  | -------- |
| Random Flip | True |
| Random Crop | 8% ~ 100% |
| Aspect Ratio| 3/4 ~ 4/3 |
| Random PCA Lighting | 0.1 |


# Train the model

### Download pretrained models
[FishNet150](https://github.com/kevin-ssy/FishNet)(Model trained without tricks )

[MobileNet V2 Model](https://github.com/tonylins/pytorch-mobilenet-v2)

**move them to  ./checkpoints/pre-trainedModels/**


### 1.train FishNet150

> nohup python main.py --config="cfgs/fishnet150-32-5train.yaml" --b 32 --lr 0.01 --every-decay 30 DIR --fl-gamma 2 >> fishnet150-train.log &
###  2.train MobileNet V2

> nohup python main.py --config="cfgs/mobilenetv2.yaml" --b 32 --lr 0.01 DIR --every-decay 40 --fl-gamma 2 >> mobilenetv2-bs32-train.log &

###  3.train FNet54
> nohup python main.py --config="cfgs/FeatherNet54-32.yaml" DIR --every-decay 60 -b 32 --lr 0.01 --fl-gamma 3 >>FNet54-bs32-train.log &

###  4.train FNet54-SE
> nohup python main.py --config="cfgs/FeatherNet54-se-64.yaml" --b 64 --lr 0.01 DIR --every-decay 60 --fl-gamma 3 >> FNet54-se-bs64-train.log &


## How to create a  submission file
example:
> python main.py --config="cfgs/shufflenetv2.yaml" --resume ./checkpoints/shufflenetv2_bs32/_34_best.pth.tar --val-save True

## cfgs/config.yaml
This file specifies the path to the train, test, model, and output directories.

This is the only place that specifies the path to these directories.
Any code that is doing I/O uses the appropriate base paths from config.yaml
Note: If you are using the docker container, then you do not need to change the paths in this file.

# Ensemble
```
run submission/EnsembledCode.ipynb
```
**notice：**Choose a few models with large differences in prediction results

# Serialized copy of the trained model
You can download my artifacts folder which I used to generate my final submissions: Available Soon

>[1] ChaLearn Face Anti-spoofing Attack Detection Challenge@CVPR2019,[link](https://competitions.codalab.org/competitions/20853?secret_key=ff0e7c30-e244-4681-88e4-9eb5b41dd7f7)

>[2] Shifeng Zhang, Xiaobo Wang, Ajian Liu, Chenxu Zhao, Jun Wan, Sergio Escalera, Hailin Shi, Zezheng Wang, Stan Z. Li, " CASIA-SURF: A Dataset and Benchmark for Large-scale Multi-modal Face Anti-spoofing ", arXiv, 2018 [PDF](https://arxiv.org/abs/1812.00408)
