from .lstm import LSTM
from .lstm import LSTM
from .transformer import get_transformer


def get_net(net_name, dataset, tokenizer):
    name_trans = {
        'gpt': 'gpt2',
        'bert': 'bert-base-uncased',
    }
    if net_name in name_trans:
        net_name = name_trans[net_name]

    vocab_size = len(tokenizer)
    num_labels = dataset['train'].features['label'].num_classes
    if net_name == 'lstm':
        return LSTM(vocab_size, num_labels)
    else:
        return get_transformer(net_name, vocab_size, num_labels)
