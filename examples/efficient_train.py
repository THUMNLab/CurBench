import argparse

from curbench.algorithms import EfficientTrainTrainer


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='lenet')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()


trainer = EfficientTrainTrainer(
    data_name=args.data,
    net_name=args.net,
    gpu_index=args.gpu,
    num_epochs=args.epochs,
    random_seed=args.seed,
    grow_epochs=args.epochs,
)
trainer.fit()
trainer.evaluate()