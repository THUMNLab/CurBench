import os
import time
import torch

from ..datasets.text import get_dataset, get_tokenizer, get_metric, convert_dataset
from ..backbones.text import get_net
from ..utils import set_random, create_log_dir, get_logger



class TextClassifier():
    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed, algorithm_name, 
                 data_prepare, model_prepare, data_curriculum, model_curriculum, loss_curriculum):
        self.random_seed = random_seed
        self.data_prepare = data_prepare
        self.model_prepare = model_prepare
        self.data_curriculum = data_curriculum
        self.model_curriculum = model_curriculum
        self.loss_curriculum = loss_curriculum

        set_random(self.random_seed)
        self._init_dataloader(data_name, net_name)
        self._init_model(data_name, net_name, gpu_index, num_epochs)
        self._init_logger(algorithm_name, data_name, net_name, num_epochs, random_seed)


    def _init_dataloader(self, data_name, net_name):
        self.dataset = get_dataset(data_name) # dict: {train, valid, test}
        self.metric, self.metric_name = get_metric(data_name)
        self.tokenizer = get_tokenizer(net_name)

        dataset = convert_dataset(data_name, self.dataset, self.tokenizer)
        self.train_loader = torch.utils.data.DataLoader(
            dataset['train'], batch_size=50, shuffle=True, pin_memory=True)
        if data_name == 'mnli':
            self.valid_loader = [torch.utils.data.DataLoader(
                dataset[x], batch_size=50, pin_memory=True) for x in ['validation_matched', 'validation_mismatched']]
            self.test_loader = [torch.utils.data.DataLoader(
                dataset[x], batch_size=50, pin_memory=True) for x in ['test_matched', 'test_mismatched']]
        else:
            self.valid_loader = [torch.utils.data.DataLoader(
                dataset['validation'], batch_size=50, pin_memory=True)]
            self.test_loader = [torch.utils.data.DataLoader(
                dataset['test'], batch_size=50, pin_memory=True)]

        self.data_prepare(self.train_loader)                            # curriculum part


    def _init_model(self, data_name, net_name, gpu_index, num_epochs):
        self.net = get_net(net_name, self.dataset, self.tokenizer)
        self.device = torch.device('cuda:%d' % (gpu_index) \
            if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        self.epochs = num_epochs
        self.optimizer = torch.optim.SGD(
            self.net.parameters(), lr=1.0)                              # for lstm
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-5)
        # self.optimizer = torch.optim.AdamW(
        #   self.net.parameters(), lr=2e-5)                             # for pretrained bert, gpt
        # self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
        if self.net.num_labels == 1: # data_name == 'stsb'
            self.criterion = torch.nn.MSELoss(reduction='none')
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

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
        best_metrics = [0.0] * len(self.valid_loader)

        for epoch in range(self.epochs):
            t = time.time()
            total = 0
            train_loss = 0.0
            predictions, references = [], []

            loader = self.data_curriculum()                             # curriculum part
            net = self.model_curriculum()                               # curriculum part

            net.train()
            for step, data in enumerate(loader):
                inputs = {k: v.to(self.device) for k, v in data.items() 
                          if k not in ['labels', 'indices']}
                labels = data['labels'].to(self.device)
                indices = data['indices'].to(self.device)

                self.optimizer.zero_grad()
                outputs = net(**inputs)[0] # logits, (hidden_states), (attentions)
                loss = self.loss_curriculum(outputs, labels, indices)   # curriculum part
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                self.optimizer.step()

                train_loss += loss.item() * len(labels)
                total += len(labels)
                references += labels.tolist()
                if net.num_labels == 1:
                    predictions += outputs.squeeze()
                else:
                    predictions += outputs.argmax(dim=1).tolist()
            
            self.lr_scheduler.step()
            train_metric = self.metric.compute(predictions=predictions, references=references)[self.metric_name]
            self.logger.info(
                '[%3d]  Train data = %7d  Train %s = %.4f  Loss = %.4f  Time = %.2fs'
                % (epoch + 1, total, self.metric_name, train_metric, train_loss / total, time.time() - t))

            if (epoch + 1) % self.log_interval == 0:
                valid_metrics = [self._valid(valid_loader) for valid_loader in self.valid_loader]
                if valid_metrics[0] > best_metrics[0]:   # for mnli, choose best acc in validation_matched
                    best_metrics[:] = valid_metrics[:]
                    torch.save(net.state_dict(), os.path.join(self.log_dir, 'net.pkl'))
                for valid_loader, valid_metric, best_metric in zip(self.valid_loader, valid_metrics, best_metrics):
                    self.logger.info(
                        '[%3d]  Valid data = %7d  Valid %s = %.4f  Best Valid %s = %.4f' 
                        % (epoch + 1, len(valid_loader.dataset), self.metric_name, valid_metric, self.metric_name, best_metric))
            

    def _valid(self, loader):
        predictions, references = [], []

        self.net.eval()
        with torch.no_grad():
            for data in loader:
                inputs = {k: v.to(self.device) for k, v in data.items() 
                          if k not in ['labels', 'indices']}
                labels = data['labels'].to(self.device)
                outputs = self.net(**inputs)[0]
                references += labels.tolist()
                if self.net.num_labels == 1:
                    predictions += outputs.squeeze()
                else:
                    predictions += outputs.argmax(dim=1).tolist()
        return self.metric.compute(predictions=predictions, references=references)[self.metric_name]


    def fit(self):
        set_random(self.random_seed)
        self._train()


    def evaluate(self, net_dir=None):
        self._load_best_net(net_dir)
        for valid_loader, test_loader in zip(self.valid_loader, self.test_loader):
            valid_metric = self._valid(valid_loader)
            test_metric = self._valid(test_loader)
            self.logger.info('Best Valid %s = %.4f and Final Test %s = %.4f' 
                            % (self.metric_name, valid_metric, self.metric_name, test_metric))
        return test_metric


    def export(self, net_dir=None):
        self._load_best_net(net_dir)
        return self.net


    def _load_best_net(self, net_dir):
        if net_dir is None: net_dir = self.log_dir
        net_file = os.path.join(net_dir, 'net.pkl')
        assert os.path.exists(net_file), 'Assert Error: the net file does not exist'
        self.net.load_state_dict(torch.load(net_file))