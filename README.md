# Komorebi
Komorebi is a general machine learning platform that interfaces with common frameworks like Tensorflow and Keras. The platform provides interfaces for 3 critical parts of a machine learning pipeline

1. Dataset Generation 
2. Model Specification
3. Model Training

The design philosophy is to modularize machine learning to iterate faster. In japanese, komorebi means sunlight through the trees. In our context, statistical insights represent sunlight and the data represents trees.

### Dataset Generation
The first step in running the attention model is to convert raw deepsea data into data structures that are more amenable for doing deep learning. To do that, we will create a dataset object (see `ml_types`) in the preprocessing library that will be needed for our prediction pipeline. 

