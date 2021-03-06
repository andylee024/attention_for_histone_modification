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


class AnnotationExtractor(object):
    """Annotation extractor for attention models."""

    def __init__(self, model, layer_name):
        """Initialize extractor.

        :param model:
            Keras model used to extract annotation vectors.
        :param layer_name:
            Str. Name of layer for which to extract convolutional layers.
        """
        self._model = model
        self._layer_name = layer_name
        self._extractor = Model(inputs=self._model.input, outputs=self._model.get_layer(layer_name).output)

    def extract_annotation(self, sequence):
        """Return annotation vector corresponding to sequence.
        
        :param sequence:
            Numpy array representing genetic sequence.
        :return:
            Feature corresponding to sequence.
        """
        annotation = self._extractor.predict(sequence)
        return annotation.flatten()

    def extract_annotation_batch(self, sequence_batch):
        """Return annotation vectors corresponding to batch of sequences.
        
        :param sequence_batch:
            Numpy array representing genetic sequence batch.
            Shape: (batch_size, sequence_length, vocabulary_size)
        :return:
            Feature corresponding to sequence.
            Shape: (batch_size, annotation_vector_size) 
        """
        return self._extractor.predict_on_batch(sequence_batch)


def get_trained_danq_model(danq_weights_path):
    """Get trained danq model by initializing with weights.

    :param danq_weights_path: 
        HDF5 weights file in Keras with which to initialize DANQ model.
    :return:
        DANQ model with weights.
    """
    danq_model = _build_danq_model()
    danq_model.load_weights(danq_weights_path)
    return danq_model


def _build_danq_model(numLabels=919, numConvFilters=320, poolingDropout=0.2, brnnDropout=0.5, learningRate=0.001):
    """Build danq model with specified parameters."""

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

def main():
    model = _build_danq_model()
    print model.summary()

if __name__ == "__main__":
    main()
