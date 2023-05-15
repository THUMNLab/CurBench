from transformers import AutoTokenizer

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
    if net_name == 'lstm':
        net = LSTM(vocab_size, dataset.num_classes)
    else:
        net = get_transformer(net_name, vocab_size, dataset.num_classes)
    net.__setattr__('name', net_name)
    net.__setattr__('num_classes', dataset.num_classes)
    return net


def get_tokenizer(net_name):
    tokenizer_dict = {
        'gpt': 'gpt2',
        'bert': 'bert-base-uncased',
        'lstm': 'bert-base-uncased',
    }
    assert net_name in tokenizer_dict, \
        'Assert Error: net_name should be in ' + str(list(tokenizer_dict.keys()))

    tokenizer_name = tokenizer_dict[net_name]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if 'gpt' in tokenizer_name:     # for gpt
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer
