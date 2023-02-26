import os
import argparse

from curbench.algorithms import DDSTrainer


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='lenet')
parser.add_argument('--epochs', type=int, default=100000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--catnum', type=int, default=10)
parser.add_argument('--epsilon', type=float, default=1e-3)
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

trainer = DDSTrainer(
    data_name=args.data,
    net_name=args.net,
    num_epochs=args.epochs,
    random_seed=args.seed,
    catnum=args.catnum,
    epsilon=args.epsilon,
    lr=args.lr,
)
trainer.fit()
trainer.evaluate()