import evaluate

from .glue import get_glue_dataset
from .utils import LabelNoise, ClassImbalanced, SplitedDataset


data_dict = {
    'cola': 'matthews_correlation', 
    'sst2': 'accuracy', 
    'mrpc': 'f1',
    'qqp':  'f1', 
    'stsb': 'spearmanr',
    'mnli': 'accuracy', 
    'qnli': 'accuracy', 
    'rte':  'accuracy', 
    'wnli': 'accuracy'
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
    'wnli': [0, 1]
}

def get_dataset(data_name, tokenizer):
    assert not ('noise' in data_name and 'imbalance' in data_name), \
        'Assert Error: only support one setting from [standard, noise, imbalance]'
    
    # noise setting
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
        noise_ratio = None

    # imbalance setting
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
            assert False, 'Assert Error: data_name shoule be \
                [dataset]-imbalance-[mode]-[dominant_labels]-[dominant_ratio]-[dominant_minor_floor]-[exp_mu]'
    else:
        imbalance_mode = None

    # get standard, noisy or imbalanced dataset
    assert data_name in data_dict, 'Assert Error: data_name should be in ' + str(list(data_dict.keys()))
    raw_dataset, converted_dataset = get_glue_dataset(data_name, tokenizer)
    # converted_dataset = SplitedDataset(converted_dataset)
    if noise_ratio:
        label_range = task_text_label_range_map[data_name]
        converted_dataset = LabelNoise(converted_dataset, noise_ratio, label_range, label_int=(data_name != 'stsb'))
    if imbalance_mode:
        converted_dataset = ClassImbalanced(converted_dataset, imbalance_mode, imbalance_dominant_labels,
                                            imbalance_dominant_ratio, imbalance_dominant_minor_floor, imbalance_exp_mu)
    return raw_dataset, converted_dataset


def get_metric(data_name):
    # allow data name format: [data]-[noise/imbalance]-[args]
    data_name = data_name.split('-')[0]
    assert data_name in data_dict, \
            'Assert Error: data_name should be in ' + str(list(data_dict.keys()))
    return evaluate.load('glue', data_name), data_dict[data_name]
