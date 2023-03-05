import evaluate
from transformers import AutoTokenizer

from .glue import get_glue_dataset, convert_dataset


data_dict = {
    'cola': 'matthews_correlation', 
    'sst2': 'accuracy', 
    'mrpc': 'f1',
    'qqp':  'f1', 
    'stsb': 'spearmanr',
    'mnli': 'accuracy', 
    'qnli': 'accuracy', 
    'rte':  'accuracy', 
    'wnli': 'accuracy', 
    'ax':   'accuracy',
}


def get_dataset(data_name):
    assert data_name in data_dict, \
            'Assert Error: data_name should be in ' + str(list(data_dict.keys()))
    return get_glue_dataset(data_name)


def get_metric(data_name):
    assert data_name in data_dict, \
            'Assert Error: data_name should be in ' + str(list(data_dict.keys()))
    return evaluate.load('glue', data_name), data_dict[data_name]


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
