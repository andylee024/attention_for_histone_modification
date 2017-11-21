
class TrainerConfiguration(object):
    """Configuration object for trainer."""

    def __init__(self, name, epochs, batch_size, save_directory, save_frequency):
        """Initialize trainer config.

        The trainer configuration specifies the training procedure, which encapsulates
        aspects like training batch size as well as saving model checkpoints.
       
        :param name: name of training procedure
        :param save_directory: directory where model checkpoints are saved
        :param save_frequency: save after this many epochs

        :param epochs: number of training epochs
        :param batch_size: number of training examples to process per iteration
        :param save_directory: location to save model
        """
        self.name = name
        self.save_directory = save_directory
        self.save_frequency = save_frequency

        self.epochs = epochs
        self.batch_size = batch_size
