# import os
# import time
# import torch

# from ..datasets.text import get_corpus, batchify
# from ..backbones.text import get_net, SplitCrossEntropyLoss
# from ..utils import get_logger, set_random



# class LanguageModel():
#     def __init__(self, data_name, net_name, num_epochs, random_seed,
#                  algorithm_name, data_prepare, model_prepare, data_curriculum, 
#                  model_curriculum, loss_curriculum):
#         self.random_seed = random_seed
#         set_random(self.random_seed)

#         self.data_prepare = data_prepare
#         self.model_prepare = model_prepare
#         self.data_curriculum = data_curriculum
#         self.model_curriculum = model_curriculum
#         self.loss_curriculum = loss_curriculum

#         self._init_dataloader(data_name)
#         self._init_model(data_name, net_name, num_epochs)
#         self._init_logger(algorithm_name, data_name, net_name, num_epochs, random_seed)


#     def _init_dataloader(self, data_name):
#         self.corpus = get_corpus(data_name)

#         self.train_data = batchify(corpus.train, batch_size=20)
#         self.valid_data = batchify(corpus.valid, batch_size=10)
#         self.test_data = batchify(corpus.test, batch_size=1)

#         # train_dataset, valid_dataset, test_dataset = \
#         #     get_dataset('./data', data_name)

#         # self.train_loader = torch.utils.data.DataLoader(
#         #     train_dataset, batch_size=100, shuffle=True,
#         #     num_workers=2, pin_memory=True)
#         # self.valid_loader = torch.utils.data.DataLoader(
#         #     valid_dataset, batch_size=100, shuffle=False,
#         #     num_workers=2, pin_memory=True)
#         # self.test_loader = torch.utils.data.DataLoader(
#         #     test_dataset, batch_size=100, shuffle=False,
#         #     num_workers=2, pin_memory=True)

#         # self.data_prepare(self.train_loader)


#     def _init_model(self, data_name, net_name, num_epochs):
#         ntokens = len(self.corpus.dictionary)
#         splits = []
#         if ntokens > 500000:
#             # One Billion
#             # This produces fairly even matrix mults for the buckets:
#             # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
#             splits = [4200, 35000, 180000]
#         elif ntokens > 75000:
#             # WikiText-103
#             splits = [2800, 20000, 76000]
#         print('Using', splits)
#         self.criterion = SplitCrossEntropyLoss(400, splits=splits, verbose=False)

#         self.net = RNNModel(net_name, ntokens, 400, 1150, 3, 0.25, 0.25, 0.4, 0.1, 0.5, True)
#         self.device = torch.device('cuda:0' \
#             if torch.cuda.is_available() else 'cpu')
#         self.net.to(self.device)
#         self.criterion.to(device)

#         self.epochs = num_epochs
#         self.lr = 30
#         self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=1.2e-6)
#         # self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
#         # self.optimizer = torch.optim.SGD(
#         #     self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
#         # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         #     self.optimizer, T_max=self.epochs, eta_min=1e-6)

#         # self.model_prepare(
#         #     self.net, self.device, self.epochs, 
#         #     self.criterion, self.optimizer, self.lr_scheduler)

    
#     def _init_logger(self, algorithm_name, data_name, 
#                      net_name, num_epochs, random_seed):
#         self.log_interval = 1

#         log_info = '%s-%s-%s-%d-%d-%s' % (
#             algorithm_name, data_name, net_name, num_epochs, random_seed,
#             time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
#         )
#         self.log_dir = os.path.join('./runs', log_info)
#         if not os.path.exists('./runs'): os.mkdir('./runs')
#         if not os.path.exists(self.log_dir): os.mkdir(self.log_dir)
#         else: print('The directory %s has already existed.' % (self.log_dir))
        
#         log_file = os.path.join(self.log_dir, 'train.log')
#         self.logger = get_logger(log_file, log_info)


#     def repackage_hidden(self, h):
#         """Wraps hidden states in new Tensors,
#         to detach them from their history."""
#         if isinstance(h, torch.Tensor):
#             return h.detach()
#         else:
#             return tuple(repackage_hidden(v) for v in h)

#     def _train_one_epoch(self):
#         # Turn on training mode which enables dropout.
#         if args.model == 'QRNN': model.reset()
#         total_loss = 0
#         start_time = time.time()
#         ntokens = len(corpus.dictionary)
#         hidden = model.init_hidden(args.batch_size)
#         batch, i = 0, 0
#         while i < train_data.size(0) - 1 - 1:
#             bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
#             # Prevent excessively small or negative sequence lengths
#             seq_len = max(5, int(np.random.normal(bptt, 5)))
#             # There's a very small chance that it could select a very long sequence length resulting in OOM
#             # seq_len = min(seq_len, args.bptt + 10)

#             lr2 = optimizer.param_groups[0]['lr']
#             optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
#             model.train()
#             data, targets = get_batch(train_data, i, args, seq_len=seq_len)

#             # Starting each batch, we detach the hidden state from how it was previously produced.
#             # If we didn't, the model would try backpropagating all the way to start of the dataset.
#             hidden = repackage_hidden(hidden)
#             optimizer.zero_grad()

#             output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
#             raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

#             loss = raw_loss
#             # Activiation Regularization
#             if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
#             # Temporal Activation Regularization (slowness)
#             if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
#             loss.backward()

#             # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
#             if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
#             optimizer.step()

#             total_loss += raw_loss.data
#             optimizer.param_groups[0]['lr'] = lr2
#             if batch % args.log_interval == 0 and batch > 0:
#                 cur_loss = total_loss.item() / args.log_interval
#                 elapsed = time.time() - start_time
#                 self.logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
#                         'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
#                     epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
#                     elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
#                 total_loss = 0
#                 start_time = time.time()
#             ###
#             batch += 1
#             i += seq_len


#     def _train(self):
#         # Loop over epochs.
#         best_val_loss = []
#         stored_loss = 100000000

#         # At any point you can hit Ctrl + C to break out of training early.
#         try:
#             for epoch in range(1, 500+1):
#                 epoch_start_time = time.time()
#                 _train_one_epoch()
#                 if 't0' in optimizer.param_groups[0]:
#                     tmp = {}
#                     for prm in model.parameters():
#                         tmp[prm] = prm.data.clone()
#                         prm.data = optimizer.state[prm]['ax'].clone()

#                     val_loss2 = evaluate(val_data)
#                     self.logger.info('-' * 89)
#                     self.logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
#                         'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
#                             epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
#                     self.logger.info('-' * 89)

#                     if val_loss2 < stored_loss:
#                         model_save(args.save)
#                         self.logger.info('Saving Averaged!')
#                         stored_loss = val_loss2

#                     for prm in model.parameters():
#                         prm.data = tmp[prm].clone()

#                 else:
#                     val_loss = evaluate(val_data, eval_batch_size)
#                     self.logger.info('-' * 89)
#                     self.logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
#                         'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
#                     epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
#                     self.logger.info('-' * 89)

#                     if val_loss < stored_loss:
#                         model_save(args.save)
#                         self.logger.info('Saving model (new best validation)')
#                         stored_loss = val_loss

#                     if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
#                         self.logger.info('Switching to ASGD')
#                         optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

#                     if epoch in args.when:
#                         self.logger.info('Saving model before learning rate decreased')
#                         model_save('{}.e{}'.format(args.save, epoch))
#                         self.logger.info('Dividing learning rate by 10')
#                         optimizer.param_groups[0]['lr'] /= 10.

#                     best_val_loss.append(val_loss)

#         except KeyboardInterrupt:
#             print('-' * 89)
#             print('Exiting from training early')

#         best_acc = 0.0

#         for epoch in range(self.epochs):
#             t = time.time()
#             total = 0
#             correct = 0
#             train_loss = 0.0

#             loader = self.data_curriculum(self.train_loader)    # curriculum part
#             net = self.model_curriculum(self.net)               # curriculum part

#             net.train()
#             for step, data in enumerate(loader):
#                 inputs = data[0].to(self.device)
#                 labels = data[1].to(self.device)
#                 indices = data[2].to(self.device)

#                 self.optimizer.zero_grad()
#                 outputs = net(inputs)
#                 loss = self.loss_curriculum(                    # curriculum part
#                     self.criterion, outputs, labels, indices
#                 )
#                 loss.backward()
#                 self.optimizer.step()

#                 train_loss += loss.item()
#                 _, predicted = outputs.max(dim=1)
#                 correct += predicted.eq(labels).sum().item()
#                 total += labels.shape[0]
            
#             self.lr_scheduler.step()
#             self.logger.info(
#                 '[%3d]  Train data = %6d  Train Acc = %.4f  Loss = %.4f  Time = %.2f'
#                 % (epoch + 1, total, correct / total, train_loss / (step + 1), time.time() - t)
#             )

#             if (epoch + 1) % self.log_interval == 0:
#                 valid_acc = self._valid(self.valid_loader)
#                 if valid_acc > best_acc:
#                     best_acc = valid_acc
#                     torch.save(net.state_dict(), os.path.join(self.log_dir, 'net.pkl'))
#                 self.logger.info(
#                     '[%3d]  Valid data = %6d  Valid Acc = %.4f' 
#                     % (epoch + 1, len(self.valid_loader.dataset), valid_acc)
#                 )


#     def _evaluate(data_source, batch_size=10):
#         # Turn on evaluation mode which disables dropout.
#         model.eval()
#         if args.model == 'QRNN': model.reset()
#         total_loss = 0
#         ntokens = len(corpus.dictionary)
#         hidden = model.init_hidden(batch_size)
#         for i in range(0, data_source.size(0) - 1, args.bptt):
#             data, targets = get_batch(data_source, i, args, evaluation=True)
#             output, hidden = model(data, hidden)
#             total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
#             hidden = repackage_hidden(hidden)
#         return total_loss.item() / len(data_source)


#     def _valid(self, loader):
#         total = 0
#         correct = 0

#         self.net.eval()
#         with torch.no_grad():
#             for data in loader:
#                 inputs = data[0].to(self.device)
#                 labels = data[1].to(self.device)

#                 outputs = self.net(inputs)
#                 _, predicted = torch.max(outputs, dim=1)
#                 total += labels.shape[0]
#                 correct += predicted.eq(labels).sum().item()
#         return correct / total


#     def fit(self):
#         set_random(self.random_seed)
#         self._train()


#     def evaluate(self, net_dir=None):
#         self._load_best_net(net_dir)
#         test_acc = self._valid(self.test_loader)
#         self.logger.info('Final Test Acc = %.4f' % (test_acc))
#         return test_acc


#     def export(self, net_dir=None):
#         self._load_best_net(net_dir)
#         return self.net


#     def _load_best_net(self, net_dir):
#         if net_dir is None: net_dir = self.log_dir
#         net_file = os.path.join(net_dir, 'net.pkl')
#         assert os.path.exists(net_file), \
#             'Assert Error: the net file does not exist'
#         self.net.load_state_dict(torch.load(net_file))


#     def model_save(fn):
#         with open(fn, 'wb') as f:
#             torch.save([model, criterion, optimizer], f)

#     def model_load(fn):
#         global model, criterion, optimizer
#         with open(fn, 'rb') as f:
#             model, criterion, optimizer = torch.load(f)