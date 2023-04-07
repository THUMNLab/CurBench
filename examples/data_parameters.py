import argparse

from curbench.algorithms import DataParametersTrainer


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='lenet')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--init_class_param', type=float, default=1.0)
parser.add_argument('--lr_class_param', type=float, default=0.1)
parser.add_argument('--wd_class_param', type=float, default=0.0)
parser.add_argument('--init_data_param', type=float, default=1.0)
parser.add_argument('--lr_data_param', type=float, default=0.1)
parser.add_argument('--wd_data_param', type=float, default=0.0)
args = parser.parse_args()


trainer = DataParametersTrainer(
    data_name=args.data,
    net_name=args.net,
    gpu_index=args.gpu,
    num_epochs=args.epochs,
    random_seed=args.seed,
    init_class_param=args.init_class_param,
    lr_class_param=args.lr_class_param,
    wd_class_param=args.wd_class_param,
    init_data_param=args.init_data_param,
    lr_data_param=args.lr_data_param,
    wd_data_param=args.wd_data_param,
)
trainer.fit()
trainer.evaluate()