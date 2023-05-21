import argparse

from curbench.algorithms import BaseTrainer , AdaptiveTrainer


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='lenet')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--pace_p', type=float, default=0.1)
parser.add_argument('--pace_q', type=float, default=1.2)
parser.add_argument('--pace_r', type=int, default=15)
parser.add_argument('--inv', type=int, default=20)
parser.add_argument('--alpha', type=float, default=0.7)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--gamma_decay', type=float, default=None)
parser.add_argument('--bottom_gamma', type=float, default=0.1)
parser.add_argument('--teacher_net', type=str, default='lenet')
parser.add_argument('--teacher_epochs', type=int, default=200)
parser.add_argument('--teacher_dir', type=str, default=None)
args = parser.parse_args()


pretrainer = BaseTrainer(
    data_name=args.data,
    # net_name=args.teacher_net,
    net_name=args.net,
    gpu_index=args.gpu,
    num_epochs=args.teacher_epochs,
    random_seed=args.seed,
)
# if args.teacher_dir is None:
#     pretrainer.fit()
args.teacher_dir = 'runs/base-%s-%s-%d-%d' % (args.data, args.net, args.epochs, 42)
pretrainer.evaluate(args.teacher_dir)
teacher_net = pretrainer.export(args.teacher_dir)


trainer = AdaptiveTrainer(
    data_name=args.data,
    net_name=args.net,
    gpu_index=args.gpu,
    num_epochs=args.epochs,
    random_seed=args.seed,
    pace_p=args.pace_p,
    pace_q=args.pace_q,
    pace_r=args.pace_r,
    inv=args.inv,
    alpha=args.alpha,
    gamma=args.gamma,
    gamma_decay=args.gamma_decay,
    bottom_gamma=args.bottom_gamma,
    pretrained_net=teacher_net,
)
trainer.fit()
trainer.evaluate()