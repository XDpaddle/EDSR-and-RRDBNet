  # EDSR_paddle

Paddle 复现版本

## 数据集

分类之后训练集用于训练SR模块
https://aistudio.baidu.com/aistudio/datasetdetail/106261

## 训练模型

链接：https://pan.baidu.com/s/1SwjPpF-SzoP_GhLqdAHwag?pwd=1234 
提取码：1234 

## 训练步骤
### train sr
```bash
python train.py -opt config/train/train_EDSR_x2.yml
python train.py -opt config/train/train_EDSR_x3.yml
python train.py -opt config/train/train_EDSR_x4.yml
```
## 测试步骤
```bash
python test.py -opt config/test/test_EDSR_x2.yml
python test.py -opt config/test/test_EDSR_x3.yml
python test.py -opt config/test/test_EDSR_x4.yml
```

  # RRDBNet_paddle

Paddle 复现版本

## 数据集

分类之后训练集用于训练SR模块
https://aistudio.baidu.com/aistudio/datasetdetail/106261

## 训练模型
链接：https://pan.baidu.com/s/1SwjPpF-SzoP_GhLqdAHwag?pwd=1234 
提取码：1234 
## 训练步骤
### train sr
```bash
python train.py -opt config/train/train_RRDBNet_x2.yml
```
## 测试步骤
```bash
python test.py -opt config/test/test_RRDBNet_x2.yml
```
