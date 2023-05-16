import datasets


def convert_dataset(data_name, tokenizer, dataset, max_length=128):

    def convert_with_tokenizer(batch):
        # Either encode single sentence or sentence pairs
        if len(text_fields) > 1:
            texts_or_text_pairs = list(zip(batch[text_fields[0]], batch[text_fields[1]]))
        else:
            texts_or_text_pairs = batch[text_fields[0]]
        # Tokenize the text/text pairs
        features = tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            padding='max_length',   # fix sequence length for shuffle sample
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        # Rename label to labels to make it easier to pass to model forward
        features['labels'] = batch['label']
        return features

    def convert_with_vocab(batch):
        # TODO: with torchtext.vocab
        return batch

    task_text_field_map = {
        'cola': ['sentence'],
        'sst2': ['sentence'],
        'mrpc': ['sentence1', 'sentence2'],
        'qqp':  ['question1', 'question2'],
        'stsb': ['sentence1', 'sentence2'],
        'mnli': ['premise', 'hypothesis'],
        'qnli': ['question', 'sentence'],
        'rte':  ['sentence1', 'sentence2'],
        'wnli': ['sentence1', 'sentence2']
    }
    text_fields = task_text_field_map[data_name]

    loader_columns = [
        'datasets_idx',
        'input_ids',
        'token_type_ids',
        'attention_mask',
        'start_positions',
        'end_positions',
        'labels'
    ]

    converted_dataset = {}
    for key in dataset.keys():
        converted_dataset[key] = dataset[key].map(
            convert_with_tokenizer, batched=True, remove_columns=['label'],
        )
        columns = [c for c in converted_dataset[key].column_names if c in loader_columns]
        converted_dataset[key].set_format(type="torch", columns=columns)
    return converted_dataset


def get_glue_dataset(data_name, tokenizer):
    raw_dataset = datasets.load_dataset('glue', data_name)
    raw_dataset.__setattr__('name', data_name)
    if data_name == 'stsb': 
        raw_dataset.__setattr__('num_classes', 6)
    else:
        raw_dataset.__setattr__('num_classes', raw_dataset['train'].features['label'].num_classes)
    converted_dataset = convert_dataset(data_name, tokenizer, raw_dataset)
    return raw_dataset, converted_dataset
