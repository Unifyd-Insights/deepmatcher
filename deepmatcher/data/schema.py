from .field import MatchingField
import logging

logger = logging.getLogger(__name__)


class ExampleSchema(object):
    def __init__(self,
                 header,
                 id_attr,
                 label_attr,
                 left_prefix,
                 right_prefix,
                 ignore_columns=[],
                 tokenize='nltk',
                 lower=True,
                 include_lengths=True):
        """
        Checks that:
        * There is a label column
        * There is an ID column
        * All columns except the label and ID columns, and ignored columns start with either
            the left table attribute prefix or the right table attribute prefix.
        * The number of left and right table attributes are the same.
        
        Create field metadata, i.e., attribute processing specification for each
        attribute.

        This includes fields for label and ID columns.

        Returns:
            list(tuple(str, MatchingField)): A list of tuples containing column name
                (e.g. "left_address") and corresponding :class:`~data.MatchingField` pairs,
                in the same order that the columns occur in the CSV file.
        """
        # assert id_attr in header
        if label_attr:
            assert label_attr in header

        for attr in header:
            if attr not in (id_attr,
                            label_attr) and attr not in ignore_columns:
                if not attr.startswith(left_prefix) and not attr.startswith(
                        right_prefix):
                    raise ValueError(
                        'Attribute ' + attr +
                        ' is not a left or a right table '
                        'column, not a label or id and is not ignored. Not sure '
                        'what it is...')

        num_left = sum(attr.startswith(left_prefix) for attr in header)
        num_right = sum(attr.startswith(right_prefix) for attr in header)

        assert num_left == num_right, "left,right attributes mismatch"

        text_field = MatchingField(lower=lower,
                                   tokenize=tokenize,
                                   init_token='<<<',
                                   eos_token='>>>',
                                   batch_first=True,
                                   include_lengths=include_lengths)
        numeric_field = MatchingField(sequential=False,
                                      preprocessing=int,
                                      use_vocab=False)
        id_field = MatchingField(sequential=False, use_vocab=False, id=True)

        fields = []
        for attr in header:
            if attr == id_attr:
                fields.append((attr, id_field))
            elif attr == label_attr:
                fields.append((attr, numeric_field))
            elif attr in ignore_columns:
                fields.append((attr, None))
            else:
                fields.append((attr, text_field))

        self.fields = dict(fields)