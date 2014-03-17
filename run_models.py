#!/usr/bin/env python
import argparse

from ..ml_utils.parse import parse_otu_matrix, load_object_from_file

def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--data-matrix', help='Input data matrix', required=True)
    args.add_argument('-e', '--model-file', help='Models file (.pkl)', default='models.pkl')
    args.add_argument('-p', '--preprocessor-file', help='Preprocessor file (.pkl)', default='prep.pkl')
    args.add_argument('-o', '--output-prefix', help='Output file prefix', default='predictions_')
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()

    sample_ids, matrix = parse_otu_matrix(args.data_matrix)
    models = load_object_from_file(args.model_file)
    data_prep = load_object_from_file(args.preprocessor_file)

    for model_name, model in models:
        predictions = model.predict(matrix)
        model_name = model_name.replace(' ', '_')
        output = open(args.output_prefix + model_name + '.txt', 'w')
        for p in predictions:
            output.write('%s\n' % (str(p)))
        output.close() 

