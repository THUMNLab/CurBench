from .lstm import LSTM
from .transformer import get_transformer


def get_net(net_name, dataset):
    name_trans = {
        'gpt': 'gpt2',
        'bert': 'bert-base-uncased',
    }

    if net_name in name_trans:
        net_name = name_trans[net_name]
    num_classes = dataset['train'].features['label'].num_classes

    if net_name == 'lstm':
        return LSTM(num_classes)
    else:
        return get_transformer(net_name, num_classes)
