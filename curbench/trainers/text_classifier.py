import os
import time
import torch
import evaluate

from ..datasets.text import get_dataset, get_tokenizer, convert_dataset
from ..backbones.text import get_net
from ..utils import get_logger, set_random, create_log_dir


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
        self.dataset = get_dataset(data_name)     # dict: {train, valid, test}
        self.tokenizer = get_tokenizer(net_name)

        dataset = convert_dataset(data_name, self.dataset, self.tokenizer)
        eval_splits = [x for x in dataset.keys() if 'validation' in x]
        self.train_loader = torch.utils.data.DataLoader(
            dataset['train'], batch_size=50, pin_memory=True)
        if len(eval_splits) == 1:
            self.valid_loader = torch.utils.data.DataLoader(
                dataset['validation'], batch_size=50, pin_memory=True)
            self.test_loader = torch.utils.data.DataLoader(
                dataset['test'], batch_size=50, pin_memory=True)
        else:
            self.valid_loader = [torch.utils.data.DataLoader(
                dataset[x], batch_size=50, pin_memory=True) for x in eval_splits]
            self.test_loader = [torch.utils.data.DataLoader(
                dataset[x], batch_size=50, pin_memory=True) for x in eval_splits]


    def _init_model(self, data_name, net_name, gpu_index, num_epochs):
        self.net = get_net(net_name, self.dataset, self.tokenizer)
        print(sum(p.numel() for p in self.net.parameters() if p.requires_grad))
        self.device = torch.device('cuda:%d' % (gpu_index) \
            if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        self.epochs = num_epochs
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=2e-5)
        self.metric = evaluate.load('glue', data_name)
        if self.net.num_labels == 1:    # data_name == 'stsb'
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()


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
            train_loss = 0.0
            predictions, references = [], []

            self.net.train()
            for step, data in enumerate(self.train_loader):
                inputs = {k: v.to(self.device) for k, v in data.items() if k != 'labels'}
                labels = data['labels'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net(**inputs)[0]  # (loss), logits, (hidden_states), (attentions)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * len(labels)
                total += len(labels)
                predictions += outputs.argmax(dim=1).tolist()
                references += labels.tolist()
            
            train_acc = self.metric.compute(predictions=predictions, references=references)['accuracy']
            self.logger.info(
                '[%3d]  Train data = %7d  Train Acc = %.4f  Loss = %.4f  Time = %.2fs'
                % (epoch + 1, total, train_acc, train_loss / total, time.time() - t))

            if (epoch + 1) % self.log_interval == 0:
                valid_acc = self._valid(self.valid_loader)
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    torch.save(self.net.state_dict(), os.path.join(self.log_dir, 'net.pkl'))
                self.logger.info(
                    '[%3d]  Valid data = %7d  Valid Acc = %.4f  Best Valid Acc = %.4f' 
                    % (epoch + 1, len(self.valid_loader.dataset), valid_acc, best_acc))
            

    def _valid(self, loader):
        predictions, references = [], []

        self.net.eval()
        with torch.no_grad():
            for data in loader:
                inputs = {k: v.to(self.device) for k, v in data.items() if k != 'labels'}
                labels = data['labels'].to(self.device)
                outputs = self.net(**inputs)[0]
                predictions += outputs.argmax(dim=1).tolist()
                references += labels.tolist()
        return self.metric.compute(predictions=predictions, references=references)['accuracy']


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