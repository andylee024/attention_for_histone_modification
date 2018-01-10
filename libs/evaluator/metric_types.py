"""Types for computing and storing metrics."""

class validation_point(object):
    """Validation point of single example for single task problem."""

    def __init__(self, classification, probability_prediction, label):
        """Initialize structure.

        :param classification: binary classification 
        :param probability_prediction: probability associated with classification
        :param label: ground-truth label
        """
        self.classification = classification
        self.probability_prediction = probability_prediction
        self.label = label


class multitask_validation_point(object):
    """Validation point of a single example for for multitask problem."""

    def __init__(self, classifications, probability_predictions, labels):
        """Initialize structure.

        :param classifications: sequence of binary classifications
        :param probability_predictions: sequence of probabilities associated with classifications
        :param labels: ground-truth labels
        """
        assert len(classifications) == len(probability_predictions)
        assert len(classifications) == len(labels)
        assert len(probability_predictions) == len(labels)

        self.classifications = classifications
        self.probability_predictions = probability_predictions
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    @property
    def single_task_validation_points(self):
        """Convert multitask validation data to single task validation data."""
        return [validation_point(classification=c, probability_prediction=p, label=l) 
                for (c, p, l) in zip(self.classifications, self,probability_predictions, self.labels)]


class task_metrics(object):
    """Struct containing metrics for a single prediction task."""

    def __init__(
            self, 
            positive_examples,
            negative_examples,
            positive_accuracy,
            negative_accuracy,
            accuracy,
            cross_entropy_loss):
        """Initialize struct
        
        :param positive_examples: number of positive examples
        :param negative_examples: number of negative examples
        :param positive_accuracy: accuracy for positive examples
        :param negative_accuracy: accuracy for negative examples
        :param accuracy: accuracy for all examples
        :param cross_entropy_loss: normalized cross-entropy loss:
        """
        self.positive_examples = positive_examples
        self.negative_examples = negative_examples

        self.positive_accuracy = positive_accuracy
        self.negative_accuracy = negative_accuracy
        self.accuracy = accuracy

        self.cross_entropy_loss = cross_entropy_loss
    @property 
    def total_examples(self):
        return self.positive_examples + self.negative_examples

