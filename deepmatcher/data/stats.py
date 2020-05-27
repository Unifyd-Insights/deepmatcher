from collections import defaultdict, Counter

import torch
import torch.nn as nn

from torchtext import data

from ..models.modules import NoMeta, Pool
from .iterator import MatchingIterator

from sklearn.decomposition import TruncatedSVD


def compute_metadata(ds, pca=False, batch_size=1024, device='cpu'):
    r"""Computes metadata about the dataset.

    Computes the following metadata about the dataset:

    * ``word_probs``: For each attribute in the dataset, a mapping from words to
      word (token) probabilities.
    * ``totals``: For each attribute in the dataset, a count of the total number of
      words present in all attribute examples.
    * ``pc``: For each attribute in the dataset, the first principal component of the
      sequence embeddings for all values of that attribute. The sequence embedding of
      an attribute value is computed by taking the weighted average of its word
      embeddings, where the weight is the soft inverse word probability. Refer
      `Arora et al. (2017) <https://openreview.net/pdf?id=SyK00v5xx>`__ for details.

    Arguments:
        pca (bool): Whether to compute the ``pc`` metadata.
    """
    metadata = {}

    # Create an iterator over the entire dataset.
    # train_iter = MatchingIterator(self,
    #                               self,
    #                               train=False,
    #                               batch_size=1024,
    #                               device='cpu',
    #                               sort_in_buckets=False)

    counter = defaultdict(Counter)

    # For each attribute, find the number of times each word id occurs in the dataset.
    # Note that word ids here also include ``UNK`` tokens, padding tokens, etc.
    for batch in data.BucketIterator(ds):
        for name in ds.all_text_fields:
            attr_input = getattr(batch, name)
            counter[name].update(attr_input.data.data.view(-1).tolist())

    word_probs = {}
    totals = {}
    for name in ds.all_text_fields:
        attr_counter = counter[name]
        total = sum(attr_counter.values())
        totals[name] = total

        field_word_probs = {}
        for word, freq in attr_counter.items():
            field_word_probs[word] = freq / total
        word_probs[name] = field_word_probs

    metadata['word_probs'] = word_probs
    metadata['totals'] = totals

    if not pca:
        return metadata

    # To compute principal components, we need to compute weighted sequence embeddings
    # for each attribute. To do so, for each attribute, we first construct a neural
    # network to compute word embeddings and take their weighted average.
    field_embed = {}
    embed = {}
    inv_freq_pool = Pool('inv-freq-avg')
    for name in ds.all_text_fields:
        field = fields[name]
        if field not in field_embed:
            vectors_size = field.vocab.vectors.shape
            embed_layer = nn.Embedding(vectors_size[0], vectors_size[1])
            embed_layer.weight.data.copy_(field.vocab.vectors)
            embed_layer.weight.requires_grad = False
            field_embed[field] = NoMeta(embed_layer)
        embed[name] = field_embed[field]

    # Create an iterator over the entire dataset.
    # train_iter = MatchingIterator(self,
    #                               self,
    #                               train=False,
    #                               batch_size=1024,
    #                               device='cpu',
    #                               sort_in_buckets=False)
    attr_embeddings = defaultdict(list)

    # Run the constructed neural network to compute weighted sequence embeddings
    # for each attribute of each example in the dataset.
    for batch in data.BucketIterator(ds, batch_size=batch_size, device=device):
        for name in ds.all_text_fields:
            attr_input = getattr(batch, name)
            embeddings = inv_freq_pool(embed[name](attr_input))
            attr_embeddings[name].append(embeddings.data.data)

    # Compute the first principal component of weighted sequence embeddings for each
    # attribute.
    pc = {}
    for name in ds.all_text_fields:
        concatenated = torch.cat(attr_embeddings[name])
        svd = TruncatedSVD(n_components=1, n_iter=7)
        svd.fit(concatenated.numpy())
        pc[name] = svd.components_[0]
    metadata['pc'] = pc

    return metadata