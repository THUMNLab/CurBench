import os
import time
import torch
import torch_geometric as pyg
from tqdm import tqdm

from ..datasets.graph import get_dataset, get_metric
from ..backbones.graph import get_net
from ..utils import set_random, create_log_dir, get_logger



class GraphClassifier():
    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed, algorithm_name, 
                 data_prepare, model_prepare, data_curriculum, model_curriculum, loss_curriculum):
        self.random_seed = random_seed
        self.data_prepare = data_prepare
        self.model_prepare = model_prepare
        self.data_curriculum = data_curriculum
        self.model_curriculum = model_curriculum
        self.loss_curriculum = loss_curriculum

        set_random(self.random_seed)
        self._init_dataloader(data_name)
        self._init_model(net_name, gpu_index, num_epochs)
        self._init_logger(algorithm_name, data_name, net_name, num_epochs, random_seed)


    def _init_dataloader(self, data_name):
        # standard:  'nci1'
        # noise:     'nci1-noise-0.4', 
        self.dataset, train_dataset, valid_dataset, test_dataset = get_dataset(data_name) # data format is a class: to shuffle and split
        self.metric, self.metric_name = get_metric(data_name)

        self.train_loader = pyg.loader.DataLoader(
            train_dataset, batch_size=50, shuffle=True, pin_memory=True, num_workers=4)
        self.valid_loader = pyg.loader.DataLoader(
            valid_dataset, batch_size=50, shuffle=False, pin_memory=True, num_workers=4)
        self.test_loader = pyg.loader.DataLoader(
            test_dataset, batch_size=50, shuffle=False, pin_memory=True, num_workers=4)

        self.data_prepare(self.train_loader)                            # curriculum part


    def _init_model(self, net_name, gpu_index, num_epochs):
        self.net = get_net(net_name, self.dataset)
        self.device = torch.device('cuda:%d' % (gpu_index) \
            if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        self.epochs = num_epochs
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

        self.model_prepare(self.net, self.device, self.epochs,          # curriculum part
            self.criterion, self.optimizer, self.lr_scheduler)


    def _init_logger(self, algorithm_name, data_name, 
                     net_name, num_epochs, random_seed):
        self.log_interval = 1
        log_info = '%s-%s-%s-%d-%d' % (
            algorithm_name, data_name, net_name, num_epochs, random_seed)
        self.log_dir = create_log_dir(log_info)
        self.logger = get_logger(os.path.join(self.log_dir, 'train.log'), algorithm_name)


    def _train(self):
        best_metric = 0.0
        for epoch in range(self.epochs):
            t = time.time()
            total = 0
            train_loss = 0.0
            predicts, targets = [], []

            loader = self.data_curriculum()                             # curriculum part
            net = self.model_curriculum()                               # curriculum part

            net.train()
            for data in tqdm(loader):
                inputs = data.to(self.device)
                labels = data.y.view(-1).to(self.device)
                indices = data.i.to(self.device)
                if len(indices) <= 1: continue # for batch norm

                self.optimizer.zero_grad()
                outputs = net(inputs)
                loss = self.loss_curriculum(outputs, labels, indices)   # curriculum part
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * data.num_graphs
                total += data.num_graphs
                predicts += outputs.softmax(dim=1).tolist() if self.dataset.name == 'ogbg-molhiv' \
                            else outputs.argmax(dim=1).tolist()
                targets += labels.tolist()

            self.lr_scheduler.step()
            train_metric = self.metric(torch.tensor(predicts), torch.tensor(targets), 
                           task="multiclass", num_classes=self.dataset.num_classes).item()
            self.logger.info(
                '[%3d]  Train Data = %6d  Train %s = %.4f  Loss = %.4f  Time = %.2fs'
                % (epoch + 1, total, self.metric_name, train_metric, train_loss / total, time.time() - t))

            if (epoch + 1) % self.log_interval == 0:
                valid_metric = self._valid(self.valid_loader)
                if valid_metric > best_metric:
                    best_metric = valid_metric
                    torch.save(net.state_dict(), os.path.join(self.log_dir, 'net.pkl'))
                self.logger.info(
                    '[%3d]  Valid Data = %6d  Valid %s = %.4f  Best Valid %s = %.4f' 
                    % (epoch + 1, len(self.valid_loader.dataset), self.metric_name, valid_metric, self.metric_name, best_metric))


    def _valid(self, loader):
        predicts, targets = [], []

        self.net.eval()
        with torch.no_grad():
            for data in tqdm(loader):
                inputs = data.to(self.device)
                labels = data.y.view(-1).to(self.device)

                outputs = self.net(inputs)
                predicts += outputs.softmax(dim=1).tolist() if self.dataset.name == 'ogbg-molhiv' \
                            else outputs.argmax(dim=1).tolist()
                targets += labels.tolist()
        return self.metric(torch.tensor(predicts), torch.tensor(targets), 
                           task="multiclass", num_classes=self.dataset.num_classes).item()
    

    def fit(self):
        set_random(self.random_seed)
        starttime = time.time()
        self._train()
        endtime = time.time()
        self.logger.info("Training Time = %d" % (endtime - starttime))
        self.logger.info("Training Mem  = %d" % (torch.cuda.max_memory_allocated(self.device)))   


    def evaluate(self, net_dir=None):
        self._load_best_net(net_dir)
        valid_metric = self._valid(self.valid_loader)
        test_metric = self._valid(self.test_loader)
        self.logger.info('Valid Data = %6d  Best Valid %s = %.4f' % (len(self.valid_loader.dataset), self.metric_name, valid_metric))
        self.logger.info('Test Data  = %6d  Final Test %s = %.4f' % (len(self.test_loader.dataset), self.metric_name, test_metric))
        return test_metric


    def export(self, net_dir=None):
        self._load_best_net(net_dir)
        return self.net


    def _load_best_net(self, net_dir):
        if net_dir is None: net_dir = self.log_dir
        net_file = os.path.join(net_dir, 'net.pkl')
        assert os.path.exists(net_file), 'Assert Error: the net file does not exist'
        self.net.load_state_dict(torch.load(net_file))
