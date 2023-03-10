# CurBench: A Curriculum Learning Benchmark

A benchmark for Curriculum Learning.

Actively under development by @THUMNLab


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

**CIFAR10** and **CIFAR100** will be downloaded automatically.

**ImageNet32** is a downsampled version of ImageNet-1k, i.e., 1281167 training images from 1000 classes and 50000 validation images with 50 images per class, but resizes all images to 32x32 pixels. It needs to be downloaded manually from the [official website](https://image-net.org/download.php). 

``` bash
CIFAR10: CurBench/data/cifar-10-batches-py/data_batch_1, ...
CIFAR100: CurBench/data/cifar-100-python/train, ...
ImageNet32: CurBench/data/imagenet32/train_data_batch_1, ...
```

### Text

**GLUE** will be downloaded automatically and it consists of **cola**, **sst2**, **mrpc**, **qqp**, **stsb**, **mnli**, **qnli**, **rte**, **wnli**, **ax**.

### Graph
**Planetoid** will be downloaded automatically and it consists of **Cora**, **CiteSeer**, **PubMed**.

**TUDataset** will be downloaded automatically and it consists of many datasets, among which we choose **NCI1**, **PROTEINS**, **COLLAB**, **DD**, **PTC_MR**, **IMDB-BINARY**.


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
python examples/base.py --data <nci1/ptc_mr/imdb-binary/> --net <gcn/gat/sage> --gpu <0/1/2/>
```


## License
We follow [Apache license](LICENSE) across the entire codebase from v0.2.
