import os
import time
import torch

from ..datasets.graph import get_dataset
from ..backbones.graph import get_net
from ..utils import set_random, create_log_dir, get_logger



class NodeClassifier():
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
        self.dataset = get_dataset(data_name)
        # self.dataset.__setattr__('dataset', self.dataset[0])
        # TODO: for data_prepare


    def _init_model(self, net_name, gpu_index, num_epochs):
        self.net = get_net(net_name, self.dataset)
        self.device = torch.device('cuda:%d' % (gpu_index) \
            if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        self.data = self.dataset[0].to(self.device)

        self.epochs = num_epochs
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=0.01, weight_decay=5e-4)
        
        # TODO: model_prepare
    
    def _init_logger(self, algorithm_name, data_name, 
                     net_name, num_epochs, random_seed):
        self.log_interval = 1
        log_info = '%s-%s-%s-%d-%d-%s' % (
            algorithm_name, data_name, net_name, num_epochs, random_seed,
            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
        self.log_dir = create_log_dir(log_info)
        self.logger = get_logger(os.path.join(self.log_dir, 'train.log'), log_info)


    def _train(self):
        best_acc = 0.0

        # TODO: data curriculum, model_curriculum and loss_curriculum
        for epoch in range(self.epochs):
            t = time.time()
            self.net.train()
            self.optimizer.zero_grad()
            masks = self.data.train_mask
            labels = self.data.y
            outputs = self.net(self.data)
            loss = self.criterion(outputs[masks], labels[masks])
            loss.backward()
            self.optimizer.step()

            train_loss = loss.item()
            predicts = outputs.argmax(dim=1)
            correct = predicts[masks].eq(labels[masks]).sum().item()
            total = masks.sum()

            self.logger.info(
                '[%3d]  Train data = %7d  Train Acc = %.4f  Loss = %.4f  Time = %.2fs'
                % (epoch + 1, total, correct / total, train_loss, time.time() - t))
            
            if (epoch + 1) % self.log_interval == 0:
                valid_acc = self._valid(self.data.val_mask)
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    torch.save(self.net.state_dict(), os.path.join(self.log_dir, 'net.pkl'))
                self.logger.info(
                    '[%3d]  Valid data = %7d  Valid Acc = %.4f  Best Valid Acc = %.4f' 
                    % (epoch + 1, self.data.val_mask.sum(), valid_acc, best_acc))


    def _valid(self, masks):
        self.net.eval()
        with torch.no_grad():
            labels = self.data.y
            outputs = self.net(self.data)
            predicts = outputs.argmax(dim=1)
            correct = predicts[masks].eq(labels[masks]).sum().item()
            total = masks.sum()
        return correct / total
    

    def fit(self):
        set_random(self.random_seed)
        self._train()


    def evaluate(self, net_dir=None):
        self._load_best_net(net_dir)
        valid_acc = self._valid(self.data.val_mask)
        test_acc = self._valid(self.data.test_mask)
        self.logger.info('Best Valid Acc = %.4f and Final Test Acc = %.4f' % (valid_acc, test_acc))
        return test_acc


    def export(self, net_dir=None):
        self._load_best_net(net_dir)
        return self.net


    def _load_best_net(self, net_dir):
        if net_dir is None: net_dir = self.log_dir
        net_file = os.path.join(net_dir, 'net.pkl')
        assert os.path.exists(net_file), 'Assert Error: the net file does not exist'
        self.net.load_state_dict(torch.load(net_file))
