from __future__ import division

import copy
import logging
import os
import pdb
from collections import Counter, defaultdict
from timeit import default_timer as timer

import pandas as pd
import pyprind
import six

from torchtext import data
from torchtext.data.example import Example
from torchtext.utils import unicode_csv_reader

from .field import MatchingField

logger = logging.getLogger(__name__)

def finalize_metadata(self):
    r"""Perform final touches to dataset metadata.

    This allows performing modifications to metadata that cannot be serialized into
    the cache.
    """

    self.orig_metadata = copy.deepcopy(self.metadata)
    for name in self.all_text_fields:
        self.metadata['word_probs'][name] = defaultdict(
            lambda: 1 / self.metadata['totals'][name],
            self.metadata['word_probs'][name])



def from_csv(fd):
    # TODO(Sid): check for all datasets to make sure the files exist and have the same schema
    a_dataset = train or validation or test
    with io.open(os.path.expanduser(os.path.join(path, a_dataset)),
                    encoding="utf8") as f:
        header = next(unicode_csv_reader(f))

    ExampleSchema(id_attr, label_attr, ignore_columns, lowercase,
                        tokenize, include_lengths))

def examples_from_file(path, schema, format='csv'):
    with open(path) as fd:
        return examples_from_iterable(fd, schema, format=format)


def examples_from_iterable(it, schema, format='csv'):
    make_example = {
        'json': Example.fromJSON,
        'dict': Example.fromdict,
        'tsv': Example.fromCSV,
        'csv': Example.fromCSV
    }[format.lower()]
    lines = 0
    if format == 'csv':
        reader = unicode_csv_reader(it)
    elif format == 'tsv':
        reader = unicode_csv_reader(it, delimiter='\t')
    else:
        reader = it
    next(reader)
    examples = [make_example(line, schema.fields) for line in reader]

    return examples


def extract_meta(trained_model):
    meta = trained_model.meta
    return dict(lowercase=meta.lowercase,
                tokenize=meta.tokenize,
                include_lengths=meta.include_lengths)


def create_from_file(column_naming, ignore_columns=[]):
    header = next(unicode_csv_reader(stream))

    if ignore_columns is None:
        ignore_columns = ignore_columns
    column_naming = dict(column_naming)
    column_naming['label'] = None
    id_attr = column_naming['id']
    label_attr = column_naming['label']
    schema = Schema(header, id_attr, label_attr, ignore_columns, lowercase,
                    tokenize, include_lengths)

    begin = timer()
    dataset_args = {'fields': fields, 'column_naming': column_naming}

    dataset = MatchingDataset(stream=stream, **dataset_args)
    dataset.all_text_fields = schema.all_text_fields
    return dataset


def split(table,
          path,
          train_prefix,
          validation_prefix,
          test_prefix,
          split_ratio=[0.6, 0.2, 0.2],
          stratified=False,
          strata_field='label'):
    """Split a pandas dataframe or CSV file into train / validation / test data sets.

    Args:
        table (pandas.Dataframe or string): The pandas dataframe or CSV file to split.
        path (string): The directory to save the train, validation and test CSV files to.
        train: Suffix to add to `path` to get the training set save path.
        validation: Suffix to add to `path` to get the validation set save path.
        test: Suffix to add to `path` to get the test set save path.
        split_ratio (List of floats): a list of 3 numbers denoting the relative sizes of
            train, test and valid splits respectively. Default is [0.6, 0.2, 0.2].
        stratified (bool): whether the sampling should be stratified.
            Default is False.
        strata_field (str): name of the examples Field stratified over.
            Default is 'label' for the conventional label field.
    """
    assert len(split_ratio) == 3

    if not isinstance(table, pd.DataFrame):
        table = pd.read_csv(table)
    if table.index.name is not None:
        table = table.reset_index()

    examples = list(table.itertuples(index=False))
    fields = [(col, None) for col in list(table)]
    dataset = data.Dataset(examples, fields)
    train, valid, test = dataset.split(split_ratio, stratified, strata_field)

    tables = (pd.DataFrame(train.examples), pd.DataFrame(valid.examples),
              pd.DataFrame(test.examples))
    prefixes = (train_prefix, validation_prefix, test_prefix)

    for i in range(len(tables)):
        tables[i].columns = table.columns
        tables[i].to_csv(os.path.join(path, prefixes[i]), index=False)


class MatchingDataset(data.Dataset):
    r"""Represents dataset with associated metadata.

    Holds all information about one split of a dataset (e.g. training set).

    Attributes:
        fields (dict): A mapping from attribute names (e.g. "left_address") to
            corresponding :class:`~data.MatchingField` objects that specify how to process
            the field.
        examples (list): A list containing all the examples (labeled tuple pairs) in this
            dataset.
        metadata (dict): Metadata about the dataset (e.g. word probabilities).
            See :meth:`~data.MatchingDataset.compute_metadata` for details.
        corresponding_field (dict): A mapping from left table attribute names
            (e.g. "left_address") to corresponding right table attribute names
            (e.g. "right_address") and vice versa.
        text_fields (dict): A mapping from canonical attribute names (e.g. "address") to
            tuples of the corresponding left and right attribute names
            (e.g. ("left_address", "right_address")).
        all_left_fields (list): A list of all left table attribute names.
        all_right_fields (list): A list of all right table attribute names.
        canonical_text_fields (list): A list of all canonical attribute names.
        label_field (str): Name of the column containing labels.
        id_field (str): Name of the column containing tuple pair ids.
    """
    def __init__(self, examples, schema, metadata=None, **kwargs):
        r"""Creates a MatchingDataset.

        Creates a MatchingDataset by performing the following, if `examples` parameter is
        not specified:

        #. Read each example (tuple pair) in specified CSV file.
        #. Preprocess example. Involves lowercasing and tokenization (unless disabled).
        #. Compute metadata. See :meth:`~data.MatchingDataset.compute_metadata` for
            details.

        If `examples` is specified, initializes MatchingDataset from given `examples`
        and `metadata` arguments.

        Arguments:
            fields (list(tuple(str, MatchingField))): A list of tuples containing column
                name (e.g. "left_address") and corresponding :class:`~data.MatchingField`
                pairs, in the same order that the columns occur in the CSV file. Tuples of
                (name, None) represent columns that will be ignored.
            column_naming (dict): A `dict` containing the following keys:
                * ``id``: The name of the tuple pair ID column.
                * ``label``: The name of the tuple pair match label column.
                * ``left``: The prefix for attribute names belonging to the left table.
                * ``right``: The prefix for attribute names belonging to the right table.
            path (str): Path to the data file. Must be specified if `examples` is None.
            format (str): The format of the data file. One of "CSV" or "TSV".
            examples (list): A list containing all the examples (labeled tuple pairs) in
                this dataset. Must be specified if `path` is None.
            metadata (dict): Metadata about the dataset (e.g. word probabilities).
                See :meth:`~data.MatchingDataset.compute_metadata` for details.
            filter_pred (callable or None): Use only examples for which
                filter_pred(example) is True, or use all examples if None.
                Default is None. This is a keyword-only parameter.
        """
        self.examples = examples
        self.metadata = metadata

        self.column_naming = column_naming

        self.corresponding_field = {}
        self.text_fields = {}

        self.all_left_fields = []
        for name, field in six.iteritems(self.fields):
            if name.startswith(
                    self.column_naming['left']) and field is not None:
                self.all_left_fields.append(name)

        self.all_right_fields = []
        for name, field in six.iteritems(self.fields):
            if name.startswith(
                    self.column_naming['right']) and field is not None:
                self.all_right_fields.append(name)

        self.canonical_text_fields = []
        for left_name in self.all_left_fields:
            canonical_name = left_name[len(self.column_naming['left']):]
            right_name = self.column_naming['right'] + canonical_name
            self.corresponding_field[left_name] = right_name
            self.corresponding_field[right_name] = left_name
            self.text_fields[canonical_name] = left_name, right_name
            self.canonical_text_fields.append(canonical_name)

        self.all_text_fields = self.all_left_fields + self.all_right_fields
        self.label_field = self.column_naming['label']
        self.id_field = self.column_naming['id']


    def get_raw_table(self):
        r"""Create a raw pandas table containing all examples (tuple pairs) in the dataset.

        To resurrect tokenized attributes, this method currently naively joins the tokens
        using the whitespace delimiter.
        """
        rows = []
        columns = list(name for name, field in six.iteritems(self.fields)
                       if field)
        for ex in self.examples:
            row = []
            for attr in columns:
                if self.fields[attr]:
                    val = getattr(ex, attr)
                    if self.fields[attr].sequential:
                        val = ' '.join(val)
                    row.append(val)
            rows.append(row)

        return pd.DataFrame(rows, columns=columns)
