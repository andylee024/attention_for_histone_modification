from komorebi.libs.dataset.dataset_config import DatasetConfiguration
from komorebi.libs.utilities.io_utils import load_pickle_object, validate_file

def create_dataset(config):
    """Create dataset from configuration file.
    
    :param config: dataset config object
    :return: dataset object
    """
    assert isinstance(config, DatasetConfiguration) 
    validate_file(config.dataset_path)
    return load_pickle_object(config.dataset_path)

