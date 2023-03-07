import os
import time
import torch
import torch_geometric as pyg

from ..datasets.graph import get_dataset, split_dataset
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
        self._init_model(data_name, net_name, gpu_index, num_epochs)
        self._init_logger(algorithm_name, data_name, net_name, num_epochs, random_seed)


    def _init_dataloader(self, data_name):
        self.dataset = get_dataset(data_name) # as a whole: to shuffle and split

        train_dataset, valid_dataset, test_dataset = split_dataset(self.dataset)
        self.train_loader = pyg.loader.DataLoader(
            train_dataset, batch_size=50, shuffle=True, pin_memory=True)
        self.valid_loader = pyg.loader.DataLoader(
            valid_dataset, batch_size=50, shuffle=False, pin_memory=True)
        self.test_loader = pyg.loader.DataLoader(
            test_dataset, batch_size=50, shuffle=False, pin_memory=True)

        self.data_prepare(self.train_loader)                            # curriculum part


    def _init_model(self, data_name, net_name, gpu_index, num_epochs):
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
        log_info = '%s-%s-%s-%d-%d-%s' % (
            algorithm_name, data_name, net_name, num_epochs, random_seed,
            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
        self.log_dir = create_log_dir(log_info)
        self.logger = get_logger(os.path.join(self.log_dir, 'train.log'), log_info)


    def _train(self):
        best_acc = 0.0
        for epoch in range(self.epochs):
            t = time.time()
            total = 0
            correct = 0
            train_loss = 0.0

            loader = self.data_curriculum()                             # curriculum part
            net = self.model_curriculum()                               # curriculum part

            net.train()
            for step, data in enumerate(loader):
                inputs = data.to(self.device)
                labels = data.y.to(self.device)
                indices = data.i.to(self.device)

                self.optimizer.zero_grad()
                outputs = net(inputs)
                loss = self.loss_curriculum(outputs, labels, indices)   # curriculum part
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * data.num_graphs
                predicts = outputs.argmax(dim=1)
                correct += predicts.eq(labels).sum().item()
                total += data.num_graphs

            self.logger.info(
                '[%3d]  Train data = %7d  Train Acc = %.4f  Loss = %.4f  Time = %.2fs'
                % (epoch + 1, total, correct / total, train_loss / total, time.time() - t))

            if (epoch + 1) % self.log_interval == 0:
                valid_acc = self._valid(self.valid_loader)
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    torch.save(net.state_dict(), os.path.join(self.log_dir, 'net.pkl'))
                self.logger.info(
                    '[%3d]  Valid data = %7d  Valid Acc = %.4f  Best Valid Acc = %.4f' 
                    % (epoch + 1, len(self.valid_loader.dataset), valid_acc, best_acc))


    def _valid(self, loader):
        total = 0
        correct = 0

        self.net.eval()
        with torch.no_grad():
            for data in loader:
                inputs = data.to(self.device)
                labels = data.y.to(self.device)

                outputs = self.net(inputs)
                predicts = outputs.argmax(dim=1)
                correct += predicts.eq(labels).sum().item()
                total += data.num_graphs
        return correct / total
    

    def fit(self):
        set_random(self.random_seed)
        self._train()


    def evaluate(self, net_dir=None):
        self._load_best_net(net_dir)
        valid_acc = self._valid(self.valid_loader)
        test_acc = self._valid(self.test_loader)
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
