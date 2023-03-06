import datasets


def get_glue_dataset(data_name):
    return datasets.load_dataset('glue', data_name)


def convert_dataset(data_name, dataset, tokenizer, max_length=128):

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
        'wnli': ['sentence1', 'sentence2'],
        'ax':   ['premise', 'hypothesis']
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

    updated_dataset = {}
    for key in dataset.keys():
        updated_dataset[key] = dataset[key].map(
            convert_with_tokenizer, batched=True, remove_columns=['label'],
        )
        columns = [c for c in updated_dataset[key].column_names if c in loader_columns]
        updated_dataset[key].set_format(type="torch", columns=columns)
    return updated_dataset
