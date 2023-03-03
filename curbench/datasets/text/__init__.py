from .glue import get_glue_dataset, convert_dataset


def get_dataset(data_name):
    if data_name in ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 
                     'mnli', 'qnli', 'rte', 'wnli', 'ax']:
        dataset = get_glue_dataset(data_name)
    else:
        raise NotImplementedError()
    return dataset
