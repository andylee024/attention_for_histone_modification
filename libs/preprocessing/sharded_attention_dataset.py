#
# Attention for Histone Modification
#

class ShardedAttentionDataset(object):
    """A sharded attention dataset that satisfies dataset API."""

    def __init__(self, index_to_dataset):
        """Initialize sharded attention dataset.

        :param index_to_dataset:
            Python dictionary where (key, value) pair is training example index and path to 
            attention dataset, respectively.
        """
        self.index_to_datasets = index_to_dataset
        self.total_examples = 0

    def get_training_examples(indices):
        """Get training examples corresponding to supplied indices.

        :param indices: List of indices corresponding to query.
        :return: List of training examples whose order maps onto supplied indices.
        """
        assert min(indices) >=0 and max(indices) < self.total_examples
        pass


class AttentionDatasetInfo(object):
    """A struct that caches information about an attention dataset.
    
    Used to optimize performance of ShardedAttentionDataset.
    """

    def __init__(self, dataset_path, indices):
        """Populate information.

        :param dataset_path:
            Absolute path of attention dataset corresponding to dataset info object.
        :param indices:
            Indices of attention dataset with respect to all samples in all datasets.
        """
        self.dataset_path = dataset_path
        self.indices = indices

    @property
    def number_of_training_examples():
        return len(self.indices)

