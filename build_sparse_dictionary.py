#!/usr/bin/env python
import argparse
from ml_utils.parse import parse_otu_matrix, save_object_to_file

# sklearn utilities
from sklearn.decomposition import DictionaryLearning
from sklearn.preprocessing import normalize

def interface():
    args = argparse.ArgumentParser()
    # Required 
    args.add_argument('-i', '--data-matrix', help='Input data matrix', required=True)
    # Optional 
    args.add_argument('-d', '--dict-file', help='Dictionary encoder file (.pkl)', default='dict.pkl')
    args.add_argument('-n', '--num-atoms', help='Desired dictionary size', default=1000, type=int)
    args.add_argument('-a', '--alpha', help='Alpha (sparsity enforcement)', default=1.0, type=float)
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()

    # Load and preprocess the data
    sample_ids, matrix = parse_otu_matrix(args.data_matrix)
    matrix = normalize(matrix)

    # Learn a dictionary 
    dict_transformer = DictionaryLearning(n_components=args.num_atoms, alpha=args.alpha)
    dict_transformer.fit(matrix)

    # Save dictionary to file  
    save_object_to_file(dict_transformer, args.dict_file)

