import warnings


def experimental(cls):
    """
    simple class decorator to inform about experimental state of class
    """
    class ExperimentalClass(cls):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            warnings.warn(f'class {cls} is experimental therefore does not expect much.')

    return ExperimentalClass
