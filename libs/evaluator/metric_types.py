

class example_score(object):
    """Struct containing scores for a single example for one prediction task."""

    def __init__(self, classification, label):
        """Initialize structure

        :param classification: classification value
        :param label: ground-truth label
        """
        self.classification = classification
        self.label = label


class metric_report(object):
    """Struct containing metrics for prediction task."""

    def __init__(self, accuracy, precision=None, recall=None):
        """Initialize struct

        :param accuracy: accuarcy score
        :param precision: precision score
        :param recall: recall score
        """
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall

