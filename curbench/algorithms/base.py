import torch
from torch.utils.data import Dataset as torchDataset
from torch.utils.data import DataLoader as torchDataLoader
from torch_geometric.data.data import Data as pygData
from torch_geometric.loader import DataLoader as pygDataLoader

from ..trainers import *



class BaseCL():
    """The base class of CL Algorithm class.

    Each CL Algorithm class has a CLDataset and five key APIs.

    Attributes:
        name: A string for the name of a CL algorithm.
        dataset: A CLDataset build by the original training dataset.
        data_size: An integer for the number of training data samples.
        batch_size: An integer for the number of a mini-batch.
        n_batches: An integer for the number of batches.
    """

    class CLDataset(torchDataset):
        """A dataset for CL Algorithm.

        It attaches the original training dataset with data index,
        which is a common strategy for data sampling or reweighting.
        """
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, index):
            """Attach data index"""
            data = self.dataset[index]
            if isinstance(data, tuple):         # data from torch.utils.data.Dataset
                data = data + (index,)          # e.g. data from cifar, imagenet
            elif isinstance(data, dict):        # data from datasets.arrow_dataset.Dataset
                data['indices'] = index         # e.g. data from glue
            elif isinstance(data, pygData):     # data from torch_geometric.datasets
                data.__setattr__('i', index)    # e.g. data from tudataset
            else:
                NotImplementedError()
            return data

        def __len__(self):
            return len(self.dataset)


    def __init__(self):
        self.name = 'base'


    def _dataloader(self, dataset, shuffle=True):
        if self.loader_type is torchDataLoader:
            return torchDataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        elif self.loader_type is pygDataLoader:
            return pygDataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        else:   # if there is any other loader class, add it
            raise NotImplementedError()


    def data_prepare(self, loader):
        """Pass training data information from Model Trainer to CL Algorithm.
        
        Initiate the CLDataset and record training data attributes.
        """
        self.loader_type = type(loader)
        self.dataset = self.CLDataset(loader.dataset)
        self.data_size = len(self.dataset)
        self.batch_size = loader.batch_size
        self.n_batches = (self.data_size - 1) // self.batch_size + 1


    def model_prepare(self, net, device, epochs, 
                      criterion, optimizer, lr_scheduler):
        """Pass model information from Model Trainer to CL Algorithm."""
        self.net = net
        self.device = device
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler


    def data_curriculum(self):
        """Measure data difficulty and schedule the training set."""
        return self._dataloader(self.dataset)


    def model_curriculum(self):
        """Schedule the model changing."""
        return self.net


    def loss_curriculum(self, outputs, labels, indices):
        """Reweight loss."""
        return torch.mean(self.criterion(outputs, labels))



class BaseTrainer():
    """The base class of CL Trainer class.

    It initiates the Model Trainer class and CL Algorithm class, 
    and provide the functions for training and evaluation.

    Attributes:
        trainer: A image classifier, language model, recommendation system, etc.
    """

    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed, cl=BaseCL()):
        """Initiate the Model Trainer according to data_name.

        If the dataset is CIFAR-10, CIFAR-100, ImageNet or their variants, the Model Trainer can be a Image Classifier.
        If the dataset is GLUE, the Model Trainer can be a Text Classifier.
        If the dataset is not a predefined one, users can create a custom Model Trainer.
        """
        trainer_dict = {
            'cifar10': ImageClassifier, 'cifar100': ImageClassifier, 'imagenet32': ImageClassifier,

            'cola': TextClassifier, 'sst2': TextClassifier, 'mrpc': TextClassifier, 'qqp': TextClassifier, 'stsb': TextClassifier, 
            'mnli': TextClassifier, 'qnli': TextClassifier, 'rte': TextClassifier, 'wnli': TextClassifier, 'ax': TextClassifier,

            # TODO: Since the data format of node classification is a graph, which can not be loaded as a dataloader,
            # TODO: we may implement curriculum learning for it in the future.
            # 'cora': NodeClassifier, 'citeseer': NodeClassifier, 'pubmed': NodeClassifier,

            'mutag': GraphClassifier, 'nci1': GraphClassifier, 'proteins': GraphClassifier, 
            'collab': GraphClassifier, 'dd': GraphClassifier, 'ptc_mr': GraphClassifier, 'imdb-binary': GraphClassifier,
        }
        assert data_name in trainer_dict, \
            'Assert Error: data_name should be in ' + str(list(trainer_dict.keys()))
        
        self.trainer = trainer_dict[data_name](
            data_name, net_name, gpu_index, num_epochs, random_seed,
            cl.name, cl.data_prepare, cl.model_prepare,
            cl.data_curriculum, cl.model_curriculum, cl.loss_curriculum,
        )
        

    def fit(self):
        return self.trainer.fit()


    def evaluate(self, net_dir=None):
        """Evaluate the net performance if given its path, else evaluate the trained net."""
        return self.trainer.evaluate(net_dir)

    
    def export(self, net_dir=None):
        """Load the net state dict if given its path, else load the trained net."""
        return self.trainer.export(net_dir)