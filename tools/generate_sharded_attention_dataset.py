#
# Attention for Histone Modification
#

import argparse
import logging
import json
import numpy as np
import os
import pickle
import shutil
import sys
from tqdm import tqdm

from attention_for_histone_modification.libs.preprocessing.extractor import AnnotationExtractor, get_trained_danq_model
from attention_for_histone_modification.libs.preprocessing.batch_processing import (
        partition_and_annotate_data, create_dataset_from_attention_partition)
from attention_for_histone_modification.libs.preprocessing.attention_types import (
        AttentionDatasetConfig, AttentionDataset, AttentionTrainingExample)
from attention_for_histone_modification.libs.preprocessing.sharded_attention_dataset import AttentionDatasetInfo 

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def main(args):


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line tool for creating a sharded attention dataset.")
    parser.add_argument("-d", "--directory", type=str, required=True, 
            help="Directory of sharded attentiond dataset info files.")
    parser.add_argument("-o", "--output", type=str, required=True, 
            help="Path to output directory for storing dataset.")
    
    args = parser.parse_args(sys.argv[1:])
    main(args)
