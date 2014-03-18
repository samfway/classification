#!/usr/bin/env python
import argparse

from ml_utils.parse import load_dataset, load_object_from_file
from ml_utils.util import convert_labels_to_int
from ml_utils.evaluation import make_evaluation_report
from ml_utils.cross_validation import get_test_sets

# sklearn classifiers 
from sklearn.ensemble import RandomForestClassifier

# sklearn utilities 
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--data-matrix', help='Input data matrix', required=True)
    args.add_argument('-o', '--output-prefix', help='Report output prefix', default='report_')
    args.add_argument('-m', '--mapping-file', help='Mapping table')
    args.add_argument('-l', '--labels-file', help='Labels file')
    args.add_argument('-c', '--metadata-category', help='Metadata category')
    args.add_argument('-v', '--metadata-value', help='Metadata value')
    args.add_argument('-k', '--k-folds', help='Number of CV folds', default=10, type=int)
    args.add_argument('-d', '--dict-file', help='Dictionary encoder file (.pkl)', default='dict.pkl')
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()
    matrix, sample_ids, labels = load_dataset(args.data_matrix, args.mapping_file, \
        args.metadata_category, args.metadata_value, args.labels_file)

    # Preprocess the data
    matrix = normalize(matrix)
    label_legend, labels = convert_labels_to_int(labels)

    # Load up all desired models
    models =  [] 
    models.append(('Random Forest', RandomForestClassifier(n_estimators=100, \
        criterion='entropy', bootstrap=True)))

    # Load up all desired performance metrics
    metrics = []   
    metrics.append(('Accuracy', accuracy_score))

    # Generate CV train/test sets
    test_sets = get_test_sets(labels, kfold=args.k_folds)

    # Perform analysis
    make_evaluation_report(models, matrix, labels, test_sets, metrics, args.output_prefix+'without.txt' )

    # Perform comparision 
    encoder = load_object_from_file(args.dict_file)    
    matrix = encoder.transform(matrix)
    make_evaluation_report(models, matrix, labels, test_sets, metrics, args.output_prefix+'with.txt' )

