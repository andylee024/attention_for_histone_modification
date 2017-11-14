
class OptimizerConfiguration(object):
    """Configuration object for instantiating optimizer."""

    def __init__(self, optimizer_type, learning_rate, momentum=None):
        """Initialize optimizer config.

        :param optimizer_type: string designating type of optimizer
        :param learning_rate: optimizer learning rate
        :param momentum: float for momentum optimizer
        """
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.momentum = momentum

