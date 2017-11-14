
class TrainerConfiguration(object):
    """Configuration object for trainer."""

    def __init__(self, epochs, batch_size):
        """Initialize trainer config.

        :param epochs: number of training epochs
        :param batch_size: number of training examples to process per iteration
        """
        self.epochs = epochs
        self.batch_size = batch_size

