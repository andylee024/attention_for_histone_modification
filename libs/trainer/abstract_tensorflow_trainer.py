import abc
import os
import tensorflow as tf
from tqdm import tqdm, trange

from komorebi.libs.trainer.abstract_trainer import AbstractTrainer
from komorebi.libs.trainer.trainer_config import TrainerConfiguration
from komorebi.libs.trainer.trainer_utils import batch_data
from komorebi.libs.utilities.io_utils import ensure_directory

TRAINED_MODEL_DIRECTORY_NAME = "trained_model"

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
        # build computational model
        graph_inputs, ops = self._build_computational_graph(model, optimizer)

        # initialization 
        init_op = tf.global_variables_initializer()
        tf.get_variable_scope().reuse_variables()

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(self._summary_directory)
        
        # initialize session and start training
        with tf.Session() as sess:
            sess.run(init_op)
            for epoch in trange(self.epochs, desc="epoch progress"):
                
                # training
                _train_epoch(dataset=dataset,
                             batch_size=self.batch_size,
                             graph_inputs=graph_inputs,
                             ops=ops, 
                             convert_training_examples=self._convert_training_examples_to_feed_dict,
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


    @abc.abstractmethod
    def _convert_training_examples_to_feed_dict(self, graph_inputs, training_examples):
        """Convert training inputs to graph inputs.

        Tensorflow models rely on passing a feed_dict into the computational graph.
        This function is responsible for translating the training examples into graph inputs.

        :param graph_inputs: dictionary mapping from string key to tf.placeholders
        :param training_examples: training example types specific to dataset.
        """
        pass


def _train_epoch(dataset, batch_size, graph_inputs, ops, convert_training_examples, writer, sess):
    """Execute training for one epoch.
    
    :param dataset: dataset to train on
    :param graph_inputs: 
    :param batch_size: size of each batch in iteration
    :param ops: dictionary of ops from computational graph
    :param convert_training_examples: function to convert training examples to feed dict.
    :param writer: tensorflow file writer for writing summaries
    :param sess: tensorflow session
    """
    training_batches, total_batches = batch_data(dataset, batch_size=batch_size)
    for training_batch in tqdm(training_batches, desc= "\t iteration progress", total=total_batches):
       _, loss, summary = sess.run(fetches=[ops['train_op'], ops['loss_op'], ops['summary_op']], 
                          feed_dict=convert_training_examples(graph_inputs, training_batch))
       writer.add_summary(summary)



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

