#!/usr/bin/env python
import argparse

from ml_utils.parse import load_dataset, load_object_from_file
from ml_utils.evaluation import evaluate_models
from ml_utils.cross_validation import get_test_sets

# sklearn classifiers 
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# sklearn utilities 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--data-matrix', help='Input data matrix', required=True)
    args.add_argument('-o', '--output-file', help='Report output file', default='report.txt')
    args.add_argument('-m', '--mapping-file', help='Mapping table')
    args.add_argument('-l', '--labels-file', help='Labels file')
    args.add_argument('-c', '--metadata-category', help='Metadata category')
    args.add_argument('-v', '--metadata-value', help='Metadata value')
    args.add_argument('-t', '--test-sets', help='Test sets (pkl)')
    args.add_argument('-k', '--k-folds', help='Number of CV folds', default=10, type=int)
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()

    matrix, sample_ids, labels, label_legend = \
        load_dataset(args.data_matrix, args.mapping_file,
                     args.metadata_category, args.metadata_value,
                     args.labels_file)

    # Preprocess the data
    scaler = StandardScaler()
    matrix = scaler.fit_transform(matrix)

    # Load up all desired models
    models =  [] 
    models.append(('Most Freq', DummyClassifier(strategy='most_frequent')))
    models.append(('Stratified', DummyClassifier(strategy='stratified')))
    #models.append(('Uniform', DummyClassifier(strategy='uniform')))
    #models.append(('Log. Regr', LogisticRegression()))
    #models.append(('Naive Bayes', GaussianNB()))
    #models.append(('SVM (RBF)', SVC(kernel='rbf', C=50, gamma=0.01)))
    models.append(('Random Forest', RandomForestClassifier(n_estimators=1000, \
        criterion='entropy', max_features=10, bootstrap=True)))
    models.append(('Extra Trees', ExtraTreesClassifier(n_estimators=1000, \
        criterion='entropy', max_features=10, bootstrap=True)))

    # Load up all desired performance metrics
    metrics = []   
    metrics.append(('Accuracy', accuracy_score))

    # Load or generate CV train/test sets
    if args.test_sets:
        test_sets = load_object_from_file(args.test_sets)
    else:
        test_sets = get_test_sets(labels, kfold=args.k_folds)

    # Perform analysis
    output_handle = open(args.output_file, 'w')
    evaluate_models(models, matrix, labels, test_sets, metrics, output_handle)

