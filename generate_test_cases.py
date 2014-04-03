#!/usr/bin/env python
import argparse

from ml_utils.parse import load_dataset, save_object_to_file
from ml_utils.cross_validation import get_test_sets

# sklearn utilities 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--data-matrix', help='Input data matrix', required=True)
    args.add_argument('-o', '--output-file', help='Test set indices', default='tests.pkl')
    args.add_argument('-m', '--mapping-file', help='Mapping table')
    args.add_argument('-l', '--labels-file', help='Labels file')
    args.add_argument('-c', '--metadata-category', help='Metadata category')
    args.add_argument('-v', '--metadata-value', help='Metadata value')
    args.add_argument('-k', '--k-folds', help='Number of CV folds', default=10, type=int)
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()
    matrix, sample_ids, labels, label_legend = \
        load_dataset(args.data_matrix, args.mapping_file,
                     args.metadata_category, args.metadata_value,
                     args.labels_file)

    # Generate CV train/test sets
    test_sets = get_test_sets(labels, kfold=args.k_folds)

    # Save test sets to file
    save_object_to_file(test_sets, args.output_file)
