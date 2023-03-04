from transformers import AutoTokenizer

from .glue import get_glue_dataset, convert_dataset


def get_dataset(data_name):
    if data_name in ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 
                     'mnli', 'qnli', 'rte', 'wnli', 'ax']:
        dataset = get_glue_dataset(data_name)
    else:
        raise NotImplementedError()
    return dataset


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
