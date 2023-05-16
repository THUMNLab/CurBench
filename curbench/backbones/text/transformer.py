from transformers import AutoConfig, AutoModelForSequenceClassification


def get_transformer(net_name, num_embeddings, num_classes):
    config = AutoConfig.from_pretrained(net_name, num_labels=num_classes)
    net = AutoModelForSequenceClassification.from_pretrained(net_name, config=config)
    # net = AutoModelForSequenceClassification.from_config(config=config)

    # Since gpt does not have <PAD>, it needs a new embedding vector
    if 'gpt' in net_name:
        net.resize_token_embeddings(num_embeddings)
        net.config.pad_token_id = num_embeddings - 1

    return net
