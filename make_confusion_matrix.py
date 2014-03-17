#!/usr/bin/env python
import argparse
from ml_utils.parse import parse_predictions_file
from ml_utils.plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-t', '--true-results-file', help='True prediction results', required=True)
    args.add_argument('-p', '--predictions-file', help='Model predictions', required=True)
    args.add_argument('-o', '--output-file', help='Output image', default='cm.pdf')
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()
    true_results = parse_predictions_file(args.true_results_file)
    predictions = parse_predictions_file(args.predictions_file)
    conf_matrix = confusion_matrix(true_results, predictions)
    labels = [ str(k+1) for k in xrange(len(conf_matrix)) ]
    plot_confusion_matrix(conf_matrix, labels, args.output_file)

