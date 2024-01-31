import argparse

from curbench.algorithms import LGLTrainer


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='lenet')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--start_rate', type=float, default=0.1)
parser.add_argument('--grow_rate', type=float, default=0.3)
parser.add_argument('--grow_interval', type=int, default=20)
parser.add_argument('--strategy', type=str, default='random')
args = parser.parse_args()


trainer = LGLTrainer(
    data_name=args.data,
    net_name=args.net,
    gpu_index=args.gpu,
    num_epochs=args.epochs,
    random_seed=args.seed,
    start_rate=args.start_rate,
    grow_rate=args.grow_rate,
    grow_interval=args.grow_interval,
    strategy=args.strategy,
)
trainer.fit()
trainer.evaluate()