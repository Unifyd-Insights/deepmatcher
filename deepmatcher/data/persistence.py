import os
import six
import logging

import torch
from timeit import default_timer as timer

from . import dataset
from . import process
from . import fields

logger = logging.getLogger(__name__)


class CacheStaleException(Exception):
    r"""Raised when the dataset cache is stale and no fallback behavior is specified.
    """


def restore_data(fields, cached_data):
    r"""Recreate datasets and related data from cache.

    This restores all datasets, metadata and attribute information (including the
    vocabulary and word embeddings for all tokens in each attribute).
    """
    datasets = []
    for d in range(len(cached_data['datafiles'])):
        metadata = None
        if d == 0:
            metadata = cached_data['train_metadata']

        examples = cached_data['examples'][d]
        if not examples:
            examples = process.examples_from_file(
                path=cached_data['datafiles'][d])
        schema = object()  #FIXME
        schema.fields = fields
        dataset = dataset.MatchingDataset(
            examples,
            schema,
            metadata=metadata,
            column_naming=cached_data['column_naming'])
        datasets.append(dataset)

    for name, field in fields:
        if name in cached_data['vocabs']:
            field.vocab = cached_data['vocabs'][name]

    return datasets


def load_cached():
       try:
            cached_data, cache_stale_cause = load_cache(
                fields_dict, datafiles, cachefile, column_naming, state_args)

            if check_cached_data and cache_stale_cause:
                if not auto_rebuild_cache:
                    raise CacheStaleException(cache_stale_cause)
                else:
                    logger.warning('Rebuilding data cache because: %s',
                                   list(cache_stale_cause))

            if not check_cached_data or not cache_stale_cause:
                datasets = restore_data(fields, cached_data)

        except IOError:
            pass


def splits(cls,
           path,
           train=None,
           validation=None,
           test=None,
           fields=None,
           embeddings=None,
           embeddings_cache=None,
           column_naming=None,
           cache=None,
           check_cached_data=True,
           auto_rebuild_cache=False,
           train_pca=False,
           **kwargs):
    r"""Create Dataset objects for multiple splits of a dataset.

        Args:
            path (str): Common prefix of the splits' file paths.
            train (str): Suffix to add to path for the train set.
            validation (str): Suffix to add to path for the validation set, or None
                for no validation set. Default is None.
            test (str): Suffix to add to path for the test set, or None for no test
                set. Default is None.
            fields (list(tuple(str, MatchingField))): A list of tuples containing column
                name (e.g. "left_address") and corresponding :class:`~data.MatchingField`
                pairs, in the same order that the columns occur in the CSV file. Tuples of
                (name, None) represent columns that will be ignored.
            embeddings (str or list): Same as `embeddings` parameter of
                :func:`~data.process`.
            embeddings_cache (str): Directory to store dowloaded word vector data.
            column_naming (dict): Same as `column_naming` paramter of `__init__`.
            cache (str): Suffix to add to path for cache file. If `None` disables caching.
            check_cached_data (bool): Verify that data files haven't changes since the
                cache was constructed and that relevant field options haven't changed.
            auto_rebuild_cache (bool): Automatically rebuild the cache if the data files
                are modified or if the field options change. Defaults to False.
            train_pca (bool): Whether to compute PCA for each attribute as part of
                dataset metadata compuatation. Defaults to False.
            filter_pred (callable or None): Use only examples for which
                filter_pred(example) is True, or use all examples if None.
                Default is None. This is a keyword-only parameter.

        Returns:
            Tuple[MatchingDataset]: Datasets for (train, validation, and test) splits in
                that order, if provided.
        """

    fields_dict = dict(fields)
    state_args = {'train_pca': train_pca}

    datasets = None
    if cache:
        datafiles = list(f for f in (train, validation, test) if f is not None)
        datafiles = [
            os.path.expanduser(os.path.join(path, d)) for d in datafiles
        ]
        cachefile = os.path.expanduser(os.path.join(path, cache))
 
    if not datasets:
        begin = timer()
        dataset_args = {
            'fields': fields,
            'column_naming': column_naming,
            **kwargs
        }

        def mkdataset(f):
            if f is None:
                return None
            fpath = os.path.join(path, f)
            examples = examples_from_file(fpath, fields)
            return cls(examples, **dataset_args)

        train_data = mkdataset(train)
        val_data = mkdataset(validation)
        test_data = mkdataset(test)
        datasets = tuple(d for d in (train_data, val_data, test_data)
                         if d is not None)

        after_load = timer()
        logger.info('Data load took: {time}s', time=(after_load - begin))

        fields_set = set(fields_dict.values())
        for field in fields_set:
            if field is not None and field.use_vocab:
                field.build_vocab(*datasets,
                                  vectors=embeddings,
                                  cache=embeddings_cache)
        after_vocab = timer()
        logger.info('Vocab construction time: {time}s',
                    time=(after_vocab - after_load))

        if train:
            datasets[0].compute_metadata(train_pca)
        after_metadata = timer()
        logger.info('Metadata computation time: {time}s',
                    time=(after_metadata - after_vocab))

        if cache:
            MatchingDataset.save_cache(datasets, fields_dict, datafiles,
                                       cachefile, column_naming, state_args)
            after_cache = timer()
            logger.info('Cache save time: {time}s',
                        time=(after_cache - after_vocab))

    if train:
        datasets[0].finalize_metadata()

        # Save additional information to train dataset.
        datasets[0].embeddings = embeddings
        datasets[0].embeddings_cache = embeddings_cache
        datasets[0].train_pca = train_pca

    # Set vocabs.
    for dataset in datasets:
        dataset.vocabs = {
            name: datasets[0].fields[name].vocab
            for name in datasets[0].all_text_fields
        }

    if len(datasets) == 1:
        return datasets[0]
    return tuple(datasets)


def state_args_compatibility(cur_state, old_state):
    errors = []
    if not old_state['train_pca'] and cur_state['train_pca']:
        errors.append('PCA computation necessary.')
    return errors


def save_cache(datasets, fields, datafiles, cachefile, column_naming,
               state_args):
    r"""Save datasets and corresponding metadata to cache.

    This method also saves as many data loading arguments as possible to help ensure
    that the cache contents are still relevant for future data loading calls. Refer
    to :meth:`~data.Dataset.load_cache` for more details.

    Arguments:
        datasets (list): List of datasets to cache.
        fields (dict): Mapping from attribute names (e.g. "left_address") to
            corresponding :class:`~data.MatchingField` objects that specify how to
            process the field.
        datafiles (list): A list of the data files.
        cachefile (str): The cache file path.
        column_naming (dict): A `dict` containing column naming conventions. See
            `__init__` for details.
        state_args (dict): A `dict` containing other information about the state under
            which the cache was created.
    """
    examples = [dataset.examples for dataset in datasets]
    train_metadata = datasets[0].metadata
    datafiles_modified = [os.path.getmtime(datafile) for datafile in datafiles]
    vocabs = {}
    field_args = {}
    reverse_fields = {}
    for name, field in six.iteritems(fields):
        reverse_fields[field] = name

    for field, name in six.iteritems(reverse_fields):
        if field is not None and hasattr(field, 'vocab'):
            vocabs[name] = field.vocab
    for name, field in six.iteritems(fields):
        field_args[name] = None
        if field is not None:
            field_args[name] = field.preprocess_args()

    data = {
        'examples': examples,
        'train_metadata': train_metadata,
        'vocabs': vocabs,
        'datafiles': datafiles,
        'datafiles_modified': datafiles_modified,
        'field_args': field_args,
        'state_args': state_args,
        'column_naming': column_naming
    }
    torch.save(data, cachefile)


def load_cache(fields, datafiles, cachefile, column_naming, state_args):
    r"""Load datasets and corresponding metadata from cache.

    This method also checks whether any of the data loading arguments have changes
    that make the cache contents invalid. The following kinds of changes are currently
    detected automatically:

    * Data filename changes (e.g. different train filename)
    * Data file modifications (e.g. train data modified)
    * Column changes (e.g. using a different subset of columns in CSV file)
    * Column specification changes (e.g. changing lowercasing behavior)
    * Column naming convention changes (e.g. different labeled data column)

    Arguments:
        fields (dict): Mapping from attribute names (e.g. "left_address") to
            corresponding :class:`~data.MatchingField` objects that specify how to
            process the field.
        datafiles (list): A list of the data files.
        cachefile (str): The cache file path.
        column_naming (dict): A `dict` containing column naming conventions. See
            `__init__` for details.
        state_args (dict): A `dict` containing other information about the state under
            which the cache was created.

    Returns:
        Tuple containing unprocessed cache data dict and a list of cache staleness
        causes, if any.

    .. warning::
        Note that if a column specification, i.e., arguments to
        :class:`~data.MatchingField` include callable arguments (e.g. lambdas or
        functions) these arguments cannot be serialized and hence will not be checked
        for modifications.
    """
    cached_data = torch.load(cachefile)
    cache_stale_cause = set()

    if datafiles != cached_data['datafiles']:
        cache_stale_cause.add('Data file list has changed.')

    datafiles_modified = [os.path.getmtime(datafile) for datafile in datafiles]
    if datafiles_modified != cached_data['datafiles_modified']:
        cache_stale_cause.add('One or more data files have been modified.')

    if set(fields.keys()) != set(cached_data['field_args'].keys()):
        cache_stale_cause.add('Fields have changed.')

    for name, field in six.iteritems(fields):
        none_mismatch = (field is None) != (cached_data['field_args'][name] is
                                            None)
        args_mismatch = False
        if field is not None and cached_data['field_args'][name] is not None:
            args_mismatch = field.preprocess_args(
            ) != cached_data['field_args'][name]
        if none_mismatch or args_mismatch:
            cache_stale_cause.add('Field arguments have changed.')
        if field is not None and not isinstance(field, fields.MatchingField):
            cache_stale_cause.add('Cache update required.')

    if column_naming != cached_data['column_naming']:
        cache_stale_cause.add('Other arguments have changed.')

    cache_stale_cause.update(
        state_args_compatibility(state_args, cached_data['state_args']))

    return cached_data, cache_stale_cause
