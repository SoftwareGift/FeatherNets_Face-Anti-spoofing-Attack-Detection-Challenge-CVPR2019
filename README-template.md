# MobileLiteNets for [Face Anti-spoofing Attack Detection Challenge@CVPR2019](https://competitions.codalab.org/competitions/20853#results)[1]
TODO: Describe what is MobileLiteNets. How we explain FeatherNet?

## Train
### Dataset 
We use 2 dataset to train. One is [CASIA-SURF Dataset](https://arxiv.org/abs/1812.00408), another is created by ourselves, named Multi-Modality Face Dataset', abbreviated as MMFD.

### How to train
TODO: data preparation, training script, where is the training model located

Commands to train the model:
####  Train MobileLiteNet54
```
python main.py --config="cfgs/MobileLiteNet54-32.yaml" --every-decay 60 -b 32 --lr 0.01 --fl-gamma 3 >>FNet54-bs32-train.log
```
####  Train MobileLiteNet54-SE
```
python main.py --config="cfgs/MobileLiteNet54-se-64.yaml" --b 64 --lr 0.01  --every-decay 60 --fl-gamma 3 >> FNet54-se-bs64-train.log
```
#### Train FeatherNetA
```
python main.py --config="cfgs/FeatherNetA-32.yaml" --b 32 --lr 0.01  --every-decay 60 --fl-gamma 3 >> MobileLiteNetA-bs32-train.log
```
#### Train FeatherNetB
```
python main.py --config="cfgs/FeatherNetB-32.yaml" --b 32 --lr 0.01  --every-decay 60 --fl-gamma 3 >> MobileLiteNetB-bs32--train.log

```

TODO: Describe where is the trained model located?

### Peformance
TODO: framework used, model size, performance data (can use the following table as performance data)
|model name | ACER|TPR@FPR=10E-2|TPR@FPR=10E-3|FP|FN|epoch|params|FLOPs|
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|MobileLiteNet54|0.00242|1.0|0.99846|32|0|41|0.57M|270.91M|
|MobileLiteNet54-se|0.00242|1.0|0.996994|32|0|69|0.57M|270.91M|
|FeatherNetA|0.00261|1.00|0.961590|19|7|51|0.35M|79.99M|
|FeatherNetB|0.00168|1.0|0.997662|20|1|48|0.35M|83.05M|

## Test
TODO: Inference time for FeatherNet and MobileLiteNet

>[1] ChaLearn Face Anti-spoofing Attack Detection Challenge@CVPR2019,[link](https://competitions.codalab.org/competitions/20853?secret_key=ff0e7c30-e244-4681-88e4-9eb5b41dd7f7)
>[2] Shifeng Zhang, Xiaobo Wang, Ajian Liu, Chenxu Zhao, Jun Wan, Sergio Escalera, Hailin Shi, Zezheng Wang, Stan Z. Li, " CASIA-SURF: A Dataset and Benchmark for Large-scale Multi-modal Face Anti-spoofing ", arXiv, 2018 [PDF](https://arxiv.org/abs/1812.00408)
