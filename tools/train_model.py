from attention_for_histone_modification.libs.model.attention_configuration import AttentionConfiguration
from attention_for_histone_modification.libs.model.attention_model import AttentionModel 
from attention_for_histone_modification.libs.model.parameter_initialization import ParameterInitializationPolicy
from attention_for_histone_modification.libs.optimizer.optimizer_config import OptimizerConfiguration
from attention_for_histone_modification.libs.optimizer.optimizer_factory import create_tf_optimizer 
from attention_for_histone_modification.libs.trainer.attention_trainer import AttentionTrainer
from attention_for_histone_modification.libs.preprocessing.utilities import load_pickle_object

def main():

    # setup
    dataset_path = "/Users/andy/Projects/biology/research/attention_for_histone_modification/data/attention_validation_dataset/sharded_attention_dataset.pkl"
    attention_config = AttentionConfiguration(batch_size=100,
                                              sequence_length=1000,
                                              vocabulary_size=4,
                                              prediction_classes=919,
                                              number_of_annotations=1,
                                              annotation_size=925,
                                              hidden_state_dimension=112)
    parameter_policy = ParameterInitializationPolicy()
    optimizer_config = OptimizerConfiguration(optimizer_type="adam", learning_rate=0.01)

    # create objects
    dataset = load_pickle_object(dataset_path)
    model = AttentionModel(attention_config=attention_config, parameter_policy=parameter_policy)
    optimizer = create_tf_optimizer(optimizer_config)
    trainer = AttentionTrainer()

    # train model
    trainer.train_model(dataset=dataset, model=model, optimizer=optimizer)

if __name__ == "__main__":
    main()


