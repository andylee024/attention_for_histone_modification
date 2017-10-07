# 
# Attention Model for Histone Modification
#
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.optimizers import RMSprop

import numpy as np

def get_danq_model(numLabels=919, numConvFilters=320, poolingDropout=0.2, brnnDropout=0.5, learningRate=0.001):
    """Return canonical danq model with default parameters.

    :return: danq model in Keras
    """

    # specify layers
    conv_layer = Conv1D(input_shape = (1000, 4),
                        padding="valid",
                        strides=1,
                        activation="relu",
                        kernel_size=26,
                        filters=numConvFilters)
    brnn = Bidirectional(LSTM(320, return_sequences=True))
    
    # specify optimizer
    optim = RMSprop(lr=learningRate, rho=0.9, epsilon=1e-08, decay=0.0)
   
    # specify network architecture
    model = Sequential()
    model.add(conv_layer)
    model.add(MaxPooling1D(pool_size=13, strides=13))
    model.add(Dropout(poolingDropout))
    model.add(brnn)
    model.add(Dropout(brnnDropout))
    model.add(Flatten())
    model.add(Dense(input_dim=75*640, units=925))
    model.add(Activation('relu'))
    model.add(Dense(units=numLabels))
    model.add(Activation('sigmoid'))

    # compile model
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
    return model

def get_trained_danq_model(danq_weights_path):
    """Get trained danq model by initializing with weights.

    :param danq_weights_path: 
        HDF5 weights file in Keras with which to initialize DANQ model.
    :return:
        DANQ model with weights.
    """
    danq_model = get_danq_model()
    danq_model.load_weights(danq_weights_path)
    return danq_model

def test_danq():
    """Simple test to test initialization of danq model."""
    danq_weights = '/Users/andy/Projects/bio_startup/research/attention_for_histone_modification/experimental/danq_weights.hdf5'
    danq_model = get_trained_danq_model(danq_weights)
    print danq_model.summary()

    dummy_training_sequence = np.zeros(shape=(1, 1000, 4))

    annotation_vector_model = Model(inputs=danq_model.input,
                                    outputs=danq_model.get_layer("dense_1").output)

    annotation_vector = annotation_vector_model.predict(dummy_training_sequence).flatten()
    assert annotation_vector.size == 925
    print "DANQ test passed..."

if __name__ == "__main__":
    test_danq()
    
