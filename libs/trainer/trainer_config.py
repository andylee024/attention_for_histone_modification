
class TrainerConfiguration(object):
    """Configuration object for trainer."""

    def __init__(self, 
                 epochs, 
                 batch_size, 
                 checkpoint_frequency):

        """Initialize trainer configuration.
       
        :param experiment_directory: directory where training results are stored
        :param checkpoint_frequency: create model checkpoint after this many epochs
        :param epochs: total training epochs
        :param batch_size: number of training examples to process per iteration in epoch
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_frequency = checkpoint_frequency

