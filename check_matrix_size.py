#!/usr/bin/env python
import argparse
from ml_utils.parse import parse_otu_matrix

def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--data-matrix', help='Input data matrix', required=True)
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()
    sample_ids, matrix = parse_otu_matrix(args.data_matrix) 
    print matrix.shape

