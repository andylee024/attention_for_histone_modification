# Attention for Histone Modification
In this repository, we implement an deep-learning attention model that is used to predict cell-type activity in histone modifications. 

## Progress
The [Attention Progress Document](https://docs.google.com/document/d/1Jbcxc8zcNPKMT17TuPbZVQDw7wNN1bQMUkOk7mi7xVQ/edit) tracks current progress on the project. It also contains some design notes about how the software will be implemented.  

### Dataset Generation
The first step in running the attention model is to convert raw deepsea data into data structures that are more amenable for doing deep learning. To do that, we will create a dataset object (see `ml_types`) in the preprocessing library that will be needed for our prediction pipeline. 

**Step 1: Extract data from raw deepsea mat files**
Deepsea data is difficult to work with so we have created a tool for extracting training sequences and labels from deepsea raw mat files. These sequences and labels are saved as numpy arrays in the output directory. 
```
python convert_deepsea_data_to_numpy.py --train [DEEPSEA TRAIN MAT FILE] -- test [DEEPSEA TEST MAT FILE] --validation [DEEPSEA VALID MAT FILE] -d [OUTPUT DIRECTORY]
```
You will see the following files 
```
/output_directory/deepsea_train_X.npy
/output_directory/deepsea_train_Y.npy
/output_directory/deepsea_test_X.npy
/output_directory/deepsea_test_Y.npy
/output_directory/deepsea_valid_X.npy
/output_directory/deepsea_valid_Y.npy
```

**Step 2: Create an attention dataset**
To create an `AttentionDataset` object, run the following tool after you have created the numpy files referenced above. 
```
python -n [NAME OF DATASET] -w [WEIGHTS TO DANQ] -l [LAYER_NAME] -x [deepsea_X.npy FILE] -y [deepsea_Y.npy FILE] - d [OUTPUT_DIRECTORY]
```
This will create a pickled dataset object that can be found 
```
/output_directory/attention_dataset_test.pkl
```
