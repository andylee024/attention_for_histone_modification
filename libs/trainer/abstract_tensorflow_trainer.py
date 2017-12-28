import abc
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm, trange

from komorebi.libs.trainer.abstract_trainer import AbstractTrainer
from komorebi.libs.trainer.trainer_config import TrainerConfiguration
from komorebi.libs.trainer.trainer_utils import batch_data
from komorebi.libs.utilities.io_utils import ensure_directory

TRAINED_MODEL_DIRECTORY_NAME = "trained_model"
SEQUENCE_SHAPE = (1000, 4)
ANNOTATION_SHAPE = (75, 320)

class AbstractTensorflowTrainer(AbstractTrainer):
    """Abstract base class to facilitate training models specific to tensorflow."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, config, experiment_directory, checkpoint_directory, summary_directory):
        """Initialize trainer.
        
        :param config: trainer_config object.
        :param experiment_directory: directory corresponding to experiment
        :param checkpont_directory: directory for storing model checkpoints
        :param summary_directory: directory for storing tf summaries
        """
        assert isinstance(config, TrainerConfiguration)
        assert os.path.isdir(experiment_directory)
        assert os.path.isdir(checkpoint_directory)
        assert os.path.isdir(summary_directory)

        self._experiment_directory = experiment_directory
        self._checkpoint_directory = checkpoint_directory
        self._summary_directory = summary_directory

        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.checkpoint_frequency = config.checkpoint_frequency
        self.model_checkpoint_path = os.path.join(self._checkpoint_directory, "model-epoch-checkpoint") 


    def train_model(self, model, dataset, optimizer):
        """Training procedure for a tensorflow model.

        This is a generic training procedure for a tensorflow model. Specifically, it 
        supports training a model and logging intermediate output to the directories
        specified below. 

        |-- base_directory
            |-- model_training_directory
                |-- checkpoints
                |-- final_model.pb
                |-- summaries
        
        :param model: model object satisfying abstract model interface
        :param dataset: dataset object satisfying abstract dataset interface
        """
        ##################
        # add dataset ops
        ##################

        # get paths to tf records
        tf_dataset_directory = "/Users/andy/Projects/biology/research/komorebi/data/attention_validation_tf_dataset"
        tf_record_paths = [os.path.join(tf_dataset_directory, tf_record) for tf_record in os.listdir(tf_dataset_directory)]
        buffer_size = 5000

        # setup new computational graph
        filenames_op = tf.placeholder(tf.string, shape=[None])
        tf_dataset = tf.data.TFRecordDataset(filenames_op)
        tf_dataset = tf_dataset.prefetch(buffer_size)
        tf_dataset = tf_dataset.map(parse_example, num_parallel_calls=6)
        tf_dataset = tf_dataset.batch(self.batch_size)
        iterator = tf_dataset.make_initializable_iterator()
        
        #
        # ORIGINAL CODE
        # 
        # build computational model
        graph_inputs, ops = self._build_computational_graph(model, optimizer)

        # initialization 
        init_op = tf.global_variables_initializer()
        tf.get_variable_scope().reuse_variables()

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(self._summary_directory)
        
        # initialize session and start training
        with tf.Session() as sess:
            # session starts here
            sess.run(init_op)
            for epoch in trange(self.epochs, desc="epoch progress"):

                # shuffle all examples and reinitialize dataset (out of memory shuffling)
                np.random.shuffle(tf_record_paths)
                sess.run(iterator.initializer, {filenames_op: tf_record_paths})

                # training
                _train_epoch(dataset=tf_dataset,
                             batch_size=self.batch_size,
                             graph_inputs=graph_inputs,
                             ops=ops, 
                             writer=writer,
                             sess=sess)

                # checkpoint saving 
                if (epoch % self.checkpoint_frequency == 0):
                    saver.save(sess=sess, save_path=self.model_checkpoint_path, global_step=epoch)

            # save trained model
            _save_trained_model(prediction_signature=model.prediction_signature, 
                                experiment_directory=self._experiment_directory, 
                                sess=sess)


    @abc.abstractmethod
    def _build_computational_graph(self, model, optimizer):
        """Construct a computational graph for training a model.

        :param model: tensorflow model to be trained
        :param optimizer: optimizer for gradient backpropogation
        :return: 
            2-tuple consisting the two dictionaries. The first dictionary contains tf.placeholders
            representing inputs to the graph. The second dictionary contains ops generated by the graph.
        """
        pass


def _train_epoch(dataset, batch_size, graph_inputs, ops, writer, sess):
    """Execute training for one epoch.
    
    :param dataset: dataset to train on
    :param graph_inputs: 
    :param batch_size: size of each batch in iteration
    :param ops: dictionary of ops from computational graph
    :param convert_training_examples: function to convert training examples to feed dict.
    :param writer: tensorflow file writer for writing summaries
    :param sess: tensorflow session
    """
    count = 0
    while True:
        try:
            data = dataset.get_next()

            # populate data placeholders
            graph_inputs['sequences'] = data['sequence']
            graph_inputs['features'] = data['annotation']
            graph_inputs['labels'] = data['label']

            _, loss, summary = sess.run(fetches=[ops['train_op'], ops['loss_op'], ops['summary_op']], 
                               feed_dict=graph_inputs)

            print "current loss for iteration {} is {}".format(count, loss)
            count += 1

            writer.add_summary(summary)

        except tf.errors.OutOfRangeError:
            return


def _save_trained_model(prediction_signature, experiment_directory, sess):
    """Save the final model.
    
    :param prediction_signature: tensorflow signature of graph specific to prediction
    :param experiment_directory: directory to save final model
    :param sess: tensorflow session
    """
    trained_model_directory = os.path.join(experiment_directory, TRAINED_MODEL_DIRECTORY_NAME)
    builder = tf.saved_model.builder.SavedModelBuilder(trained_model_directory)
    builder.add_meta_graph_and_variables(sess=sess, 
                                         tags=[tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={"predict": prediction_signature})
    model_path = builder.save()
    print "saved {}".format(model_path)


def parse_example(tf_example):
    """Parse tensorflow example"""
    
    features_map = {
        'sequence_raw': tf.FixedLenFeature([], tf.string),
        'label_raw': tf.FixedLenFeature([], tf.string),
        'annotation_raw': tf.FixedLenFeature([], tf.string)}
    
    parsed_example = tf.parse_single_example(tf_example, features_map)
    
    sequence_raw = tf.decode_raw(parsed_example['sequence_raw'], tf.uint8)
    annotation_raw = tf.decode_raw(parsed_example['annotation_raw'], tf.float32)
    
    sequence = tf.reshape(sequence_raw, SEQUENCE_SHAPE)
    label = tf.decode_raw(parsed_example['label_raw'], tf.uint8)
    annotation = tf.reshape(annotation_raw, ANNOTATION_SHAPE)
    
    return {'sequence': sequence, 'label': label, 'annotation': annotation}

