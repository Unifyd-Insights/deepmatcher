def load():
    format = format.lower()
    make_example = {
        'json': Example.fromJSON,
        'dict': Example.fromdict,
        'tsv': Example.fromCSV,
        'csv': Example.fromCSV
    }[format]

    with io.open(os.path.expanduser(path), encoding="utf8") as f:
        if format == 'csv':
            reader = unicode_csv_reader(f, **csv_reader_params)
        elif format == 'tsv':
            reader = unicode_csv_reader(f, delimiter='\t', **csv_reader_params)
        else:
            reader = f

        if format in ['csv', 'tsv'] and isinstance(fields, dict):
            if skip_header:
                raise ValueError(
                    'When using a dict to specify fields with a {} file,'
                    'skip_header must be False and'
                    'the file must have a header.'.format(format))
            header = next(reader)
            field_to_index = {f: header.index(f) for f in fields.keys()}
            make_example = partial(make_example, field_to_index=field_to_index)

        if skip_header:
            next(reader)

        examples = [make_example(line, fields) for line in reader]

    if isinstance(fields, dict):
        fields, field_dict = [], fields
        for field in field_dict.values():
            if isinstance(field, list):
                fields.extend(field)
            else:
                fields.append(field)
