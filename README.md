# CurBench: A Curriculum Learning Benchmark

A benchmark for Curriculum Learning.

Actively under development by @THUMNLab


## Environment

1. python >= 3.6

2. pytorch >= 1.9.0


## Dataset

### Vision

**CIFAR10** and **CIFAR100** will be downloaded automatically.

**ImageNet32** is a downsampled version of ImageNet-1k, i.e., 1281167 training images from 1000 classes and 50000 validation images with 50 images per class, but resizes all images to 32x32 pixels. It needs to be downloaded manually from the [official website](https://image-net.org/download.php). 

``` bash
CIFAR10: CurBench/data/cifar-10-batches-py/data_batch_1, ...
CIFAR100: CurBench/data/cifar-100-python/train, ...
ImageNet32: CurBench/data/imagenet32/train_data_batch_1, ...
```

### Text

**Penn Treebank (PTB)**, **WikiText-2 (WT2)**, and **WikiText-103 (WT103)**
``` bash
PTB: CurBench/data/penn/train.txt, ...
WT2: CurBench/data/wikitext-2/train.txt, ...
```

### Graph


## Quick Start

``` bash
# 1. clone from the repository
git clone https://github.com/zhouyw16/CurBench
cd CurBench

# 2. pip install local module: curbench
pip install -e .

# 3. prepare dataset
mkdir data

# 4. run the example code
python examples/base.py
```


## Run
```bash
# 1. vision standard
python examples/base.py --data <cifar10/cifar100/imagenet32> --net <lenet/resnet/vit> --gpus <0/1/2/...>

# 2. text standard
python examples/base.py --data <ptb/wt2/wt103> --net <lstm/qrnn/bert> --gpus <0/1/2/...>

# 3. graph standard
```


## License
We follow [Apache license](LICENSE) across the entire codebase from v0.2.
