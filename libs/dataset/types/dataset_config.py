class DatasetConfiguration(object):
    """Configuration object for datasets."""

    def __init__(self, dataset_name, examples_directory):
        """Initialize dataset configuration.

        :param dataset_name: name of dataset
        :param dataset_directory: directory containing training example files associated with dataset
        """
        self.dataset_name = dataset_name
        self.examples_directory = examples_directory
