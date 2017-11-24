
class DatasetConfiguration(object):
    """Configuration object for datasets."""

    def __init__(self, dataset_name, dataset_path):
        """Initialize dataset configuration.

        :param dataset_name: name of dataset
        :param dataset_path: path to generated dataset
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

