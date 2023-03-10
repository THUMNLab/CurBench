import argparse

from curbench.algorithms import SuperlossTrainer


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='lenet')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--tau', type=float, default=0.0)
parser.add_argument('--lam', type=float, default=1.0)
parser.add_argument('--fac', type=float, default=0.9)
args = parser.parse_args()


trainer = SuperlossTrainer(
    data_name=args.data,
    net_name=args.net,
    gpu_index=args.gpu,
    num_epochs=args.epochs,
    random_seed=args.seed,
    tau=args.tau,
    lam=args.lam,
    fac=args.fac,
)
trainer.fit()
trainer.evaluate()