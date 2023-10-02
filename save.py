from datasets import load_dataset

for config in ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']:
    dataset = load_dataset('glue', config)
    dataset.save_to_disk('/DATA/DATANAS1/zyw16/MMData/glue/%s' % config)
