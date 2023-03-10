import evaluate
from transformers import AutoTokenizer

from .glue import get_glue_dataset, convert_dataset
from .utils import LabelNoise, ClassImbalanced


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

task_text_label_range_map = {
    'cola': [0, 1],
    'sst2': [0, 1],
    'mrpc': [0, 1],
    'qqp':  [0, 1],
    'stsb': [0, 5],
    'mnli': [0, 2],
    'qnli': [0, 1],
    'rte':  [0, 1],
    'wnli': [0, 1],
    'ax':   [0, 1],
}


def get_dataset(data_name):
    assert data_name in data_dict, \
            'Assert Error: data_name should be in ' + str(list(data_dict.keys()))
    return get_glue_dataset(data_name)


def get_dataset_with_noise(data_name):
    if 'noise' in data_name:
        try:
            parts = data_name.split('-')
            data_name = parts[0]
            noise_ratio = float(parts[-1])
        except:
            assert False, 'Assert Error: data_name shoule be [dataset]-noise-[ratio]'
        assert noise_ratio >= 0.0 and noise_ratio <= 1.0, \
            'Assert Error: noise ratio should be in range of [0.0, 1.0]'
    else:
        noise_ratio = 0.0
    
    dataset = get_dataset(data_name)
    if noise_ratio > 0.0:
        label_range = task_text_label_range_map[data_name]
        label_int = False if data_name == 'stsb' else True
        dataset = LabelNoise(dataset, noise_ratio, label_range, label_int)
    return dataset


def get_dataset_with_imbalanced_class(data_name):
    if 'imbalance' in data_name:
        try:
            parts = data_name.split('-')
            data_name = parts[0]
            imbalance_mode = parts[2]
            imbalance_dominant_labels = list(eval(parts[3]))
            imbalance_dominant_ratio = float(parts[4])
            imbalance_dominant_minor_floor = int(parts[5])
            imbalance_exp_mu = float(parts[6])
        except:
            assert False, 'Assert Error: data_name shoule be [dataset]-imbalance-[mode]-[dominant_labels]-[dominant_ratio]-[dominant_minor_floor]-[exp_mu]'
    else:
        imbalance_mode = 'none'
        imbalance_dominant_labels = None
        imbalance_dominant_ratio = 1
        imbalance_dominant_minor_floor = 0
        imbalance_exp_mu = 1
    
    dataset = get_dataset(data_name)
    if imbalance_mode != 'none':
        dataset = ClassImbalanced(dataset, imbalance_mode, imbalance_dominant_labels,\
            imbalance_dominant_ratio, imbalance_dominant_minor_floor, imbalance_exp_mu)
    return dataset


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
