import argparse

from curbench.algorithms import BabyStepTrainer


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='lenet')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--start_rate', type=float, default=0.0)
parser.add_argument('--grow_rate', type=float, default=0.1)
parser.add_argument('--grow_interval', type=int, default=20)
parser.add_argument('--not_sorted', action="store_true")
args = parser.parse_args()


trainer = BabyStepTrainer(
    data_name=args.data,
    net_name=args.net,
    gpu_index=args.gpu,
    num_epochs=args.epochs,
    random_seed=args.seed,
    start_rate=args.start_rate,
    grow_rate=args.grow_rate,
    grow_interval=args.grow_interval,
    not_sorted=args.not_sorted,
)
trainer.fit()
trainer.evaluate()