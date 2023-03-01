import os
import time
import torch

from ..datasets.graph import get_dataset
from ..backbones.graph import get_net
from ..utils import get_logger, set_random


class GraphClassifier():
    def __init__(self, data_name, net_name, num_epochs, random_seed, algorithm_name, 
                 data_prepare, model_prepare, data_curriculum, model_curriculum, loss_curriculum):
        self.random_seed = random_seed
        self.data_prepare = data_prepare
        self.model_prepare = model_prepare
        self.data_curriculum = data_curriculum
        self.model_curriculum = model_curriculum
        self.loss_curriculum = loss_curriculum

        set_random(self.random_seed)
        self._init_dataloader(data_name)
        self._init_model(data_name, net_name, num_epochs)
        self._init_logger(algorithm_name, data_name, net_name, num_epochs, random_seed)


    def _init_dataloader(self, data_name):
        self.dataset = get_dataset(data_name)
        dataset = dataset.shuffle()
        train_dataset = dataset[:150]
        test_dataset = dataset[150:]
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    def _init_model(self, data_name, net_name, num_epochs):
        self.hidden_channels = 16
        self.lr = 0.01
        self.epochs = 200

        # self.net = GCN(self.dataset.num_features, self.hidden_channels, self.dataset.num_classes)
        self.net = get_net(net_name, data_name)
        self.device = torch.device('cuda:0' \
            if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        self.optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=args.lr)  # Only perform weight-decay on first convolution.
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    
    def _init_logger(self, algorithm_name, data_name, 
                     net_name, num_epochs, random_seed):
        log_info = '%s-%s-%s-%d-%d-%s' % (
            algorithm_name, data_name, net_name, num_epochs, random_seed,
            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
        self.log_dir = os.path.join('./runs', log_info)
        if not os.path.exists('./runs'): os.mkdir('./runs')
        if not os.path.exists(self.log_dir): os.mkdir(self.log_dir)
        else: print('The directory %s has already existed.' % (self.log_dir))

        self.log_interval = 1
        self.logger = get_logger(os.path.join(self.log_dir, 'train.log'), log_info)

        init_wandb(name='%s-%s' % (net_name, data_name), lr=self.lr, epochs=self.epochs,
           hidden_channels=self.hidden_channels, device=self.device)


    def _train_one_epoch(self, epoch):
        self.net.train()
        self.optimizer.zero_grad()
        out = model(self.data.x, self.data.edge_index, self.data.edge_attr)
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def _train(self):
        best_val_acc = final_test_acc = 0
        for epoch in range(1, self.epochs + 1):
            loss = self._train_one_epoch(epoch)
            train_acc, val_acc, tmp_test_acc = self._valid()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)


    def _valid(self):
        model.eval()
        with torch.no_grad():
            pred = self.net(self.data.x, self.data.edge_index, self.data.edge_attr).argmax(dim=-1)

        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
        return accs
    

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
