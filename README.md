# CurBench: A Curriculum Learning Benchmark

A benchmark for Curriculum Learning.


## Environment

1. python >= 3.7  

    [https://www.python.org/downloads/](https://www.python.org/downloads/)

2. pytorch >= 1.12

    [https://pytorch.org/](https://pytorch.org/)

3. torch_geometric

    [https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

4. other requirements

```bash
pip install -r requirements.txt 
# for chinese user, if the download speed is slow, add:
-i https://pypi.tuna.tsinghua.edu.cn/simple
```


## Dataset

### Vision

**CIFAR-10** and **CIFAR-100** will be downloaded automatically.

**Tiny-ImageNet** is  a subset of the ILSVRC2012 version of ImageNet and consists of 64 × 64 × 3 down-sampled images. It needs to be downloaded manually from the [official website](https://image-net.org/download.php). 

``` bash
CurBench
└── data
    ├── cifar-10-batches-py
    │   ├── data_batch_1
    │   ├── data_batch_2
    │   ├── ...
    │   └── test_batch
    ├── cifar-100-python
    │   ├── train
    │   ├── test
    │   └── meta
    │   └── ...
    └── tiny-imagenet-200
        ├── train
        ├── val
        └── test

# For easier data processing, we use a Tiny-ImageNet dataset utility class for pytorch: https://gist.github.com/lromor/bcfc69dcf31b2f3244358aea10b7a11b
# After the processing, the directory becomes:

CurBench
└── data
    └── tiny-imagenet-200
        ├── train_batch
        ├── val_batch
        └── ...
```

### Text

**GLUE** will be downloaded automatically and it consists of **cola**, **sst2**, **mrpc**, **qqp**, **stsb**, **mnli**, **qnli**, **rte**, ...

### Graph

**TUDataset** will be downloaded automatically and it consists of many datasets, among which we choose **MUTAG**, **PROTEINS**, **NCI1**

**OGB** will be downloaded automatically and it consists of many datasets, among which we choose **molhiv**


## Quick Start

``` bash
# 1. clone from the repository
git clone https://github.com/zhouyw16/CurBench
cd CurBench

# 2. pip install local module: curbench
pip install -e .

# 3. prepare dataset
ln -s /DATA/DATANAS1/zyw16/MMData data

# 4. run the example code
python examples/base.py
```


## Run
```bash
# 1. vision standard
python examples/base.py --data <cifar10/cifar100/imagenet32> --net <lenet/resnet/vit> --gpu <0/1/2/>

# 2. text standard
python examples/base.py --data <rte/sst2/cola/> --net <lstm/bert/gpt> --gpu <0/1/2/>

# 3. graph standard
python examples/base.py --data <nci1/ptc_mr/> --net <gcn/gat/sage> --gpu <0/1/2/>
```


## License
We follow [Apache license](LICENSE) across the entire codebase from v0.2.
