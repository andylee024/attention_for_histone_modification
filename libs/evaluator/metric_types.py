"""Types for computing and storing metrics."""

import numpy as np

class attention_interpretation(object):
    """Attention interpretation information for a single training example."""

    def __init__(self, sequence, context_probabilities):
        """Initialize structure.

        :param sequence: genetic sequence
        :param context_probabilities: context probabilities output by attention network for sequence
        """
        self.sequence = sequence
        self.context_probabilities = context_probabilities
        self._index_to_nucleotide = {0: 'a', 1: 'c', 2: 'g', 3: 't'}

    def _convert_sequence_to_string(self, sequence):
        """Connvert sequence to string representation."""
        return "".join([self._index_to_nucleotide[s] for s in sequence])

    @property
    def influence_motif(self):
        """Return influence motif and associated context probability score.
        
        :return: 2-tuple (influence_motif, influence score)
        """
        max_index = np.argmax(self.context_probabilities)
        sequence_start = max(0, max_index - 6)
        sequence_end = min(len(self.sequence), max_index + 7)

        influence_score = self.context_probabilities[max_index]
        influence_motif = self.sequence[sequence_start:sequence_end]
        influence_motif_string = self._convert_sequence_to_string(influence_motif)
        return influence_motif_string, influence_score


class validation_point(object):
    """Validation point of single example for single task problem."""

    def __init__(self, classification, probability_prediction, label, attention_interpretation_info):
        """Initialize structure.

        :param classification: binary classification 
        :param probability_prediction: probability associated with classification
        :param label: ground-truth label
        :param attention_interpretation_info: information to intrepret attention network
        """
        self.classification = classification
        self.probability_prediction = probability_prediction
        self.label = label
        self.attention_interpretation = attention_interpretation_info


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
        """Convert multitask validation point to list of single task validation points."""
        return [validation_point(classification=c, probability_prediction=p, label=l) 
                for (c, p, l) in zip(self.classifications, self.probability_predictions, self.labels)]


class task_metrics(object):
    """Struct containing metrics for a single prediction task."""

    def __init__(
            self, 
            positive_examples,
            negative_examples,
            total_accuracy,
            true_positive_rate,
            true_negative_rate,
            precision,
            recall,
            f1_score,
            auroc,
            cross_entropy):
        """Initialize struct
        
        :param positive_examples: number of positive examples
        :param negative_examples: number of negative examples
        :param total_accuracy: percentage of correctly classified examples
        :param true_positive_rate: percentage of correctly classified positive examples
        :param true_negative_rate: percentage of correctly classified negative examples
        :param precision: precision score
        :param recall: recall score
        :param f1_score: f1_score
        :param auroc: area under receiving operator curve
        :param cross_entropy_loss: normalized cross-entropy loss:
        """
        self.positive_examples = positive_examples
        self.negative_examples = negative_examples

        self.total_accuracy = total_accuracy

        self.true_positive_rate = true_positive_rate
        self.true_negative_rate = true_negative_rate
        self.false_positive_rate = 1.0 - true_negative_rate
        self.false_negative_rate = 1.0 - true_positive_rate

        self.precision = precision
        self.recall = recall 
        self.f1_score = f1_score

        self.auroc = auroc

        self.cross_entropy = cross_entropy

    def __str__(self):
        """String representation of task metrics."""
        s1 = "total_examples: {}".format(self.total_examples)
        s2 = "positive_examples: {}".format(self.positive_examples)
        s3 = "negative_examples: {}".format(self.negative_examples)
        s4 = "total_accuracy: {} ".format(self.total_accuracy)

        s5 = "true_positive_rate: {}".format(self.true_positive_rate)
        s6 = "true_negative_rate: {}".format(self.true_negative_rate)
        s7 = "false_positive_rate: {}".format(self.false_positive_rate)
        s8 = "false_negative_rate: {}".format(self.false_negative_rate)

        s9 = "precision: {}".format(self.precision)
        s10 = "recall: {}".format(self.recall)
        s11 = "f1_score: {}".format(self.f1_score)

        s12 = "AUROC: {}".format(self.auroc)
        s13 = "cross_entropy: {}".format(self.cross_entropy)

        return "\n".join([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13])

    @property 
    def total_examples(self):
        return self.positive_examples + self.negative_examples

