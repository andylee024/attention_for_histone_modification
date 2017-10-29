#
# Attention for Histone Modification
#

import datetime
import json
import os


class ShardedAttentionDataset(object):
    """A sharded attention dataset that satisfies dataset API."""

    def __init__(self, config, datasets):
        """Initialize sharded attention dataset.

        :param config: attention dataset config containing information about dataset.
        :param datasets: a list of paths to attention datasets.
        """
        self.config = config
        self.datasets = datasets


def AttentionDatasetInfo(object):
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

