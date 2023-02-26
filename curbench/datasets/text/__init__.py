import os
import torch

from .utils import Corpus


data_dict = {
    'ptb': 'data/penn',
    'wt2': 'data/wikitext-2',
    'wt103': 'data/wikitext-103'
}


def get_corpus(data_name):
    assert data_name in data_dict, \
        'Assert Error: data_name should be in ' + str(list(data_dict.keys()))
    
    data_dir = data_dict[data_name]
    data_path = os.path.join(data_dir, 'corpus.data')
    if os.path.exists(data_path):
        corpus = torch.load(data_path)
    else:
        corpus = Corpus(data_dir)
        torch.save(corpus, data_path)

    return corpus


def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data


def get_batch(source, i, args, seq_len=None):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


if __name__ == '__main__':
    corpus = get_corpus('ptb')
    print(corpus.train.shape)
    batch = batchify(corpus.train, 20)
    print(batch.shape)