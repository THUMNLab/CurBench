import argparse

from curbench.algorithms import BaseTrainer, C2FTrainer


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='lenet')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cluster_K', type=int, default=3)
parser.add_argument('--teacher_dir', type=str, default=None)
args = parser.parse_args()


pretrainer = BaseTrainer(
    data_name=args.data,
    net_name=args.net,
    gpu_index=args.gpu,
    num_epochs=args.epochs,
    random_seed=args.seed,
)
# if args.teacher_dir is None:
#     pretrainer.fit()
args.teacher_dir = 'runs/base-%s-%s-%d-%d' % (args.data.split('-')[0], args.net, args.epochs, 42)
# pretrainer.evaluate(args.teacher_dir)
teacher_net = pretrainer.export(args.teacher_dir)


trainer = C2FTrainer(
    data_name=args.data,
    net_name=args.net,
    gpu_index=args.gpu,
    num_epochs=args.epochs,
    random_seed=args.seed,
    cluster_K=args.cluster_K,
    pretrained_net=teacher_net,
)
trainer.fit()
trainer.evaluate()