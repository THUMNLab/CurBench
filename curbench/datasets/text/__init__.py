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

    # get standard, noisy or imbalanced dataset
    assert data_name in data_dict, 'Assert Error: data_name should be in ' + str(list(data_dict.keys()))
    raw_dataset, converted_dataset = get_glue_dataset(data_name, tokenizer)
    # converted_dataset = SplitedDataset(converted_dataset)
    if noise_ratio:
        label_range = task_text_label_range_map[data_name]
        converted_dataset = LabelNoise(converted_dataset, noise_ratio, label_range, label_int=(data_name != 'stsb'))
    return raw_dataset, converted_dataset


# Connect Error: huggingface.co
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef

def simple_accuracy(preds, labels):
    return float((preds == labels).mean())

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = float(f1_score(y_true=labels, y_pred=preds))
    return {
        "accuracy": acc,
        "f1": f1,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = float(pearsonr(preds, labels)[0])
    spearman_corr = float(spearmanr(preds, labels)[0])
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
    }

class GLUEMetric():
    def __init__(self, config_name):
        self.config_name = config_name

    def compute(self, predictions, references):
        predictions = np.array(predictions, dtype='int64' if self.config_name != 'stsb' else 'float32')
        references = np.array(references, dtype='int64' if self.config_name != 'stsb' else 'float32')

        if self.config_name == "cola":
            return {"matthews_correlation": matthews_corrcoef(references, predictions)}
        elif self.config_name == "stsb":
            return pearson_and_spearman(predictions, references)
        elif self.config_name in ["mrpc", "qqp"]:
            return acc_and_f1(predictions, references)
        elif self.config_name in ["sst2", "mnli", "mnli_mismatched", "mnli_matched", "qnli", "rte", "wnli", "hans"]:
            return {"accuracy": simple_accuracy(predictions, references)}
        else:
            raise KeyError(
                "You should supply a configuration name selected in "
                '["sst2", "mnli", "mnli_mismatched", "mnli_matched", '
                '"cola", "stsb", "mrpc", "qqp", "qnli", "rte", "wnli", "hans"]'
            )

def get_metric(data_name):
    # allow data name format: [data]-[noise/imbalance]-[args]
    data_name = data_name.split('-')[0]
    assert data_name in data_dict, \
            'Assert Error: data_name should be in ' + str(list(data_dict.keys()))
    # Connect Error: huggingface.co
    # return evaluate.load('glue', data_name), data_dict[data_name]
    return GLUEMetric(data_name), data_dict[data_name]
