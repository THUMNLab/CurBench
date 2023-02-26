import os
import argparse

from curbench.algorithms import RLTeacherTrainer


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='lenet')
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--policy', type=str, default='online',
                    help='online, naive, window, sampling')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

trainer = RLTeacherTrainer(
    data_name=args.data,
    net_name=args.net,
    num_epochs=args.epochs,
    random_seed=args.seed,
    policy=args.policy,
)
trainer.fit()
trainer.evaluate()