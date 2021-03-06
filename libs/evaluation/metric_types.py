"""Types for computing and storing metrics."""

import collections
import numpy as np

import komorebi.libs.utilities.constants as constants

# Struct to containing indexing information
Trace = collections.namedtuple(typename='trace', field_names=['start', 'end'])

class attention_result(object):
    """Attention result for a single training example."""

    def __init__(self, sequence, context_probabilities):
        """Initialize structure.

        :param sequence: genetic sequence
        :param context_probabilities: context probabilities output by attention network for sequence
        """
        self.sequence = sequence
        self.sequence_string = _convert_sequence_to_string(sequence)
        self.context_probabilities = context_probabilities

    def _get_motif_and_score_from_context_index(self, index):
        """Retrieve motif and scor associated with context index."""
        trace = _convert_context_index_to_sequence_trace(index)
        score = self.context_probabilities[index]
        motif = self.sequence_string[trace.start:trace.end]
        return motif, score

    @property
    def impact_motif_and_score(self):
        """Return impact motif and associated context probability score.
        
        :return: 2-tuple (influence_motif, influence score)
        """
        max_index = np.argmax(self.context_probabilities)
        return self._get_motif_and_score_from_context_index(max_index)


class validation_point(object):
    """Validation point of single example for single task problem."""

    def __init__(self, classification, probability_prediction, label, attention_result_instance):
        """Initialize structure.

        :param classification: binary classification 
        :param probability_prediction: probability associated with classification
        :param label: ground-truth label
        :param attention_result_instance: information to intrepret attention network
        """
        self.classification = classification
        self.probability_prediction = probability_prediction
        self.label = label
        self.attention_result = attention_result_instance


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


def _convert_sequence_to_string(sequence):
    """Convert one-hot vector representation of sequence to string representation.

    :param sequence: n x 4 array, where each 4-tuple along zeroth dimension encodes a nucleotide
    :return: length n string representation of sequence 
    """
    _validate_one_hot_sequence(sequence)
    integer_sequence = [np.argmax(one_hot_nucleotide) for one_hot_nucleotide in sequence]
    return "".join([constants.INDEX_TO_NUCLEOTIDE_MAP[s] for s in integer_sequence])


def _validate_one_hot_sequence(sequence):
    """Validate that one-hot sequence is correctly encoded."""
    assert sequence.shape == (1000, 4)
    for one_hot_nucleotide in sequence:
        assert sum(one_hot_nucleotide) == 1


def _convert_context_index_to_sequence_trace(context_index):
    """Get start and end sequence indices corresponding to context index.
   
    Note that this function is implementation-specific to the exact parameters of
    the DANQ model. Eventually, we will replace this function with a principled way of retrieving indices.

    :param context_index: index into one of the original context vectors
    :return: 2-tuple to index into original subsequence the context index refers to.
    """
    sequence_length = constants.DANQ_SEQUENCE_LENGTH # 1000
    kernel_size = constants.DANQ_CONV1_KERNEL_SIZE # 26
    pool_size = constants.DANQ_POOL1_STRIDE # 13
    pool_stride = constants.DANQ_POOL1_STRIDE # 13
    total_cnn_vectors = (sequence_length - kernel_size) + 1 # 975 (Note this matches (975, 320) output in Conv1 output)
    
    # 1-1 mapping of CNN output to corresponding trace
    # Each trace indexes back into the original sequence
    cnn_traces = [Trace(start=idx, end=idx+kernel_size) for idx in range(total_cnn_vectors)]
    
    # 1-1 mapping of annotation (i.e. context vector) to corresponding trace
    # Each annotation trace indexes back into a vector of CNN vectors from previous step
    annotation_starts = np.arange(start=0, stop=total_cnn_vectors, step=pool_size)
    annotation_ends = annotation_starts + (pool_stride - 1)
    annotation_index_pairs = zip(annotation_starts, annotation_ends)
    annotation_traces = [Trace(ai_start, ai_end) for (ai_start, ai_end) in annotation_index_pairs]

    ## retrieve original sequence index
    ## annotation -> cnn -> original sequence
    annotation_trace = annotation_traces[context_index]
    cnn_start_trace = cnn_traces[annotation_trace.start]
    cnn_end_trace = cnn_traces[annotation_trace.end]

    sequence_start = cnn_start_trace.start
    sequence_end = cnn_end_trace.end
    return Trace(start=sequence_start, end=sequence_end)

