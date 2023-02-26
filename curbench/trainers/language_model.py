import os
import time
import math
import numpy as np
import torch

from ..datasets.text import get_corpus, batchify, get_batch
from ..backbones.text import RNNModel, SplitCrossEntropyLoss
from ..utils import get_logger, set_random



class LanguageModel():
    def __init__(self, data_name, net_name, num_epochs, random_seed,
                 algorithm_name, data_prepare, model_prepare, data_curriculum, 
                 model_curriculum, loss_curriculum):
        self.random_seed = random_seed
        set_random(self.random_seed)

        self.data_prepare = data_prepare
        self.model_prepare = model_prepare
        self.data_curriculum = data_curriculum
        self.model_curriculum = model_curriculum
        self.loss_curriculum = loss_curriculum

        self._init_dataloader(data_name)
        self._init_model(data_name, net_name, num_epochs)
        self._init_logger(algorithm_name, data_name, net_name, num_epochs, random_seed)


    def _init_dataloader(self, data_name):
        self.batch_size = 80
        self.bptt = 70

        self.corpus = get_corpus(data_name)
        self.train_data = batchify(self.corpus.train, batch_size=self.batch_size)
        self.valid_data = batchify(self.corpus.valid, batch_size=self.batch_size)
        self.test_data = batchify(self.corpus.test, batch_size=self.batch_size)


    def _init_model(self, data_name, net_name, num_epochs):
        self.epochs = num_epochs
        self.emsize = 400
        self.nhid = 1150
        self.nlayers = 3
        self.lr = 30
        self.clip = 0.25
        self.alpha = 2.0
        self.beta = 1.0
        self.dropout = 0.4
        self.dropouth = 0.25
        self.dropouti = 0.4
        self.dropoute = 0.1
        self.wdrop = 0.5
        self.wdecay = 1.2e-6
        self.nonmono = 5

        self.ntokens = len(self.corpus.dictionary)
        splits = []
        if self.ntokens > 500000:
            # One Billion
            # This produces fairly even matrix mults for the buckets:
            # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
            splits = [4200, 35000, 180000]
        elif self.ntokens > 75000:
            # WikiText-103
            splits = [2800, 20000, 76000]
        print('Using', splits)
        self.criterion = SplitCrossEntropyLoss(400, splits=splits, verbose=False)

        self.net = RNNModel(net_name, self.ntokens, self.emsize, self.nhid, self.nlayers, 
                            self.dropout, self.dropouth, self.dropouti, self.dropoute, self.wdrop, True)
        self.device = torch.device('cuda:0' \
            if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        self.criterion.to(self.device)

        self.params = list(self.net.parameters()) + list(self.criterion.parameters())
        total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in self.params if x.size())
        print('Model total parameters:', total_params)

        self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=1.2e-6)

    
    def _init_logger(self, algorithm_name, data_name, 
                     net_name, num_epochs, random_seed):
        self.log_interval = 1

        log_info = '%s-%s-%s-%d-%d-%s' % (
            algorithm_name, data_name, net_name, num_epochs, random_seed,
            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        )
        self.log_dir = os.path.join('./runs', log_info)
        if not os.path.exists('./runs'): os.mkdir('./runs')
        if not os.path.exists(self.log_dir): os.mkdir(self.log_dir)
        else: print('The directory %s has already existed.' % (self.log_dir))
        
        log_file = os.path.join(self.log_dir, 'train.log')
        self.logger = get_logger(log_file, log_info)


    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors,
        to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)


    def _train_one_epoch(self, epoch):
        # Turn on training mode which enables dropout.
        if self.net.rnn_type == 'QRNN': self.net.reset()
        total_loss = 0
        start_time = time.time()
        hidden = self.net.init_hidden(self.batch_size)
        batch, i = 0, 0
        while i < self.train_data.size(0) - 1 - 1:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            # Prevent excessively small or negative sequence lengths
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            # There's a very small chance that it could select a very long sequence length resulting in OOM
            # seq_len = min(seq_len, args.bptt + 10)

            lr2 = self.optimizer.param_groups[0]['lr']
            self.optimizer.param_groups[0]['lr'] = lr2 * seq_len / self.bptt
            self.net.train()
            data, targets = get_batch(self.train_data, i, seq_len)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = self.repackage_hidden(hidden)
            self.optimizer.zero_grad()

            output, hidden, rnn_hs, dropped_rnn_hs = self.net(data, hidden, return_h=True)
            raw_loss = self.criterion(self.net.decoder.weight, self.net.decoder.bias, output, targets)

            loss = raw_loss
            # Activiation Regularization
            if self.alpha: loss = loss + sum(self.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            if self.beta: loss = loss + sum(self.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            if self.clip: torch.nn.utils.clip_grad_norm_(self.params, self.clip)
            self.optimizer.step()

            total_loss += raw_loss.data
            self.optimizer.param_groups[0]['lr'] = lr2
            if batch % self.log_interval == 0 and batch > 0:
                cur_loss = total_loss.item() / self.log_interval
                elapsed = time.time() - start_time
                self.logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                    epoch, batch, len(self.train_data) // self.bptt, self.optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / self.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
                total_loss = 0
                start_time = time.time()
            ###
            batch += 1
            i += seq_len


    def _train(self):
        # Loop over epochs.
        best_val_loss = []
        stored_loss = 100000000

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, self.epochs+1):
                epoch_start_time = time.time()
                self._train_one_epoch(epoch)
                if 't0' in self.optimizer.param_groups[0]:
                    tmp = {}
                    for prm in self.net.parameters():
                        tmp[prm] = prm.data.clone()
                        prm.data = self.optimizer.state[prm]['ax'].clone()

                    val_loss2 = self._valid(self.valid_data)
                    self.logger.info('-' * 89)
                    self.logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                            epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
                    self.logger.info('-' * 89)

                    if val_loss2 < stored_loss:
                        torch.save(self.net.state_dict(), os.path.join(self.log_dir, 'net.pkl'))
                        self.logger.info('Saving Averaged!')
                        stored_loss = val_loss2

                    for prm in self.net.parameters():
                        prm.data = tmp[prm].clone()

                else:
                    val_loss = self._valid(self.valid_data, self.batch_size)
                    self.logger.info('-' * 89)
                    self.logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                    self.logger.info('-' * 89)

                    if val_loss < stored_loss:
                        torch.save(self.net.state_dict(), os.path.join(self.log_dir, 'net.pkl'))
                        self.logger.info('Saving model (new best validation)')
                        stored_loss = val_loss

                    best_val_loss.append(val_loss)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')


    def _valid(self, data_source, batch_size=10):
        # Turn on evaluation mode which disables dropout.
        self.net.eval()
        if self.net.rnn_type == 'QRNN': self.net.reset()
        total_loss = 0
        hidden = self.net.init_hidden(batch_size)
        for i in range(0, data_source.size(0) - 1, self.bptt):
            data, targets = get_batch(data_source, i, self.bptt)
            output, hidden = self.net(data, hidden)
            total_loss += len(data) * self.criterion(self.net.decoder.weight, self.net.decoder.bias, output, targets).data
            hidden = self.repackage_hidden(hidden)
        return total_loss.item() / len(data_source)


    def fit(self):
        set_random(self.random_seed)
        self._train()


    def evaluate(self, net_dir=None):
        self._load_best_net(net_dir)
        test_acc = self._valid(self.test_loader)
        self.logger.info('Final Test Acc = %.4f' % (test_acc))
        return test_acc


    def export(self, net_dir=None):
        self._load_best_net(net_dir)
        return self.net


    def _load_best_net(self, net_dir):
        if net_dir is None: net_dir = self.log_dir
        net_file = os.path.join(net_dir, 'net.pkl')
        assert os.path.exists(net_file), \
            'Assert Error: the net file does not exist'
        self.net.load_state_dict(torch.load(net_file))
