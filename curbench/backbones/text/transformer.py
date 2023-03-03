from transformers import AutoConfig, AutoModelForSequenceClassification


def get_transformer(net_name, num_labels):
    config = AutoConfig.from_pretrained(net_name, num_labels=num_labels)
    return AutoModelForSequenceClassification.from_config(config)
