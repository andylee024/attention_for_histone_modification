from komorebi.libs.model.attention_configuration import AttentionConfiguration
from komorebi.libs.model.attention_model import AttentionModel 
from komorebi.libs.model.parameter_initialization import ParameterInitializationPolicy
from komorebi.libs.optimizer.optimizer_config import OptimizerConfiguration
from komorebi.libs.optimizer.optimizer_factory import create_tf_optimizer 
from komorebi.libs.trainer.attention_trainer import AttentionTrainer
from komorebi.libs.trainer.trainer_config import TrainerConfiguration
from komorebi.libs.utilities.io_utils import load_pickle_object

def main():

    # setup
    dataset_path = "/Users/andy/Projects/biology/research/komorebi/data/attention_validation_dataset/sharded_attention_dataset.pkl"
    attention_config = AttentionConfiguration(sequence_length=1000,
                                              vocabulary_size=4,
                                              prediction_classes=919,
                                              number_of_annotations=75,
                                              annotation_size=320,
                                              hidden_state_dimension=112)
    parameter_policy = ParameterInitializationPolicy()
    optimizer_config = OptimizerConfiguration(optimizer_type="adam", learning_rate=0.01)
    trainer_config = TrainerConfiguration(epochs=1, 
                                          batch_size=2000, 
                                          experiment_directory="/tmp/tf_trainer_test", 
                                          checkpoint_frequency=1)

    # create objects
    dataset = load_pickle_object(dataset_path)
    model = AttentionModel(attention_config=attention_config, parameter_policy=parameter_policy)
    optimizer = create_tf_optimizer(optimizer_config)
    trainer = AttentionTrainer(trainer_config)

    # train model
    trainer.train_model(dataset=dataset, model=model, optimizer=optimizer)

if __name__ == "__main__":
    main()


