import os
import math


## Image
methods = ['base', 'spl', 'ttcl', 'mcl', 'screener_net', 'lre', 'mw_net', 
           'dcl', 'lgl', 'dds', 'dihcl', 'superloss', 'cbs', 'c2f', 'adaptive_cl']
datasets = ['cifar10', 'cifar100', 'tinyimagenet']
models = ['lenet', 'resnet18', 'vit']
settings = ['', '-noise-0.4']
seeds = [42, 666, 777, 888, 999]
epoch = 200
gpu = 0
for setting in settings:
    for dataset in datasets:
        for model in models:
            for method in methods:
                for seed in seeds:
                    dir_name = 'runs/%s-%s%s-%s-%d-%d' % (method, dataset, setting, model, epoch, seed)
                    if not os.path.exists(dir_name):
                        cmd = 'python examples/%s.py --data %s%s --net %s --seed %d --epochs %d --gpu %d' % (method, dataset, setting, model, seed, epoch, gpu)
                        print(cmd)
                        os.system(cmd)
                    else:
                        print('Already run: %s' % dir_name)


## Text
methods = ['base', 'spl', 'ttcl', 'mcl', 'screener_net', 'lre', 'mw_net', 
           'dcl', 'dds', 'dihcl', 'superloss', 'adaptive_cl']
datasets = ['rte', 'mrpc', 'stsb', 'cola', 'sst2', 'qnli', 'qqp', 'mnli']
models = ['lstm', 'bert', 'gpt']
settings = ['', '-noise-0.4']
seeds = [42, 666, 777, 888, 999]
epochs = {'lstm': 10, 'bert': 3, 'gpt': 3}
gpu = 0
for setting in settings:
    for dataset in datasets:
        for model in models:
            for method in methods:
                for seed in seeds:
                    dir_name = 'runs/%s-%s%s-%s-%d-%d' % (method, dataset, setting, model, epochs[model], seed)
                    if not os.path.exists(dir_name):
                        cmd = 'python examples/%s.py --data %s%s --net %s --seed %d --epochs %d --gpu %d' % (method, dataset, setting, model, seed, epochs[model], gpu)
                        if method in ['spl', 'ttcl']: cmd += ' --grow_epochs %d' % (int(math.ceil(epochs[model] / 2)))
                        if method == 'mcl': cmd += ' --warm_epoch 1 --schedule_epoch 1'
                        if method == 'dihcl': cmd += ' --warm_epoch 1'
                        if method == 'adaptive_cl': cmd += ' --inv 1'
                        print(cmd)
                        os.system(cmd)
                    else:
                        print('Already run: %s' % dir_name)


## Graph
methods = ['base', 'spl', 'ttcl', 'mcl', 'screener_net', 'lre', 'mw_net', 
           'dcl', 'dds', 'dihcl', 'superloss', 'adaptive_cl']
datasets = ['mutag', 'proteins', 'nci1', 'molhiv']
models = ['gcn', 'gat', 'gin']
settings = ['', '-noise-0.4']
seeds = [42, 666, 777, 888, 999]
epoch = 200
gpu = 0
for setting in settings:
    for dataset in datasets:
        for model in models:
            for method in methods:
                for seed in seeds:
                    dir_name = 'runs/%s-%s%s-%s-%d-%d' % (method, dataset, setting, model, epoch, seed)
                    if not os.path.exists(dir_name):
                        cmd = 'python examples/%s.py --data %s%s --net %s --seed %d --epochs %d --gpu %d' % (method, dataset, setting, model, seed, epoch, gpu)
                        print(cmd)
                        os.system(cmd)
                    else:
                        print('Already run: %s' % dir_name)
