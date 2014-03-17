#!/usr/bin/env python
import argparse
from ml_utils.parse import load_dataset, save_object_to_file
from ml_utils.util import convert_labels_to_int

# sklearn classifiers
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# sklearn utilities
from sklearn.preprocessing import StandardScaler

def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--data-matrix', help='Input data matrix', required=True)
    args.add_argument('-m', '--mapping-file', help='Mapping table')
    args.add_argument('-l', '--labels-file', help='Labels file')
    args.add_argument('-c', '--metadata-category', help='Metadata category')
    args.add_argument('-v', '--metadata-value', help='Metadata value')
    args.add_argument('-e', '--model-file', help='Models file (.pkl)', default='models.pkl')
    args.add_argument('-p', '--preprocessor-file', help='Preprocessor file (.pkl)', default='prep.pkl')
    args.add_argument('-q', '--quiet', help='Do not print status messages', action='store_true', default=False)
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()

    matrix, sample_ids, labels = load_dataset(args.data_matrix, args.mapping_file, \
        args.metadata_category, args.metadata_value, args.labels_file)

    # Preprocess the data
    data_prep = StandardScaler()
    matrix = data_prep.fit_transform(matrix)
    label_legend, labels = convert_labels_to_int(labels)

    # Load up all desired models
    models =  []
    models.append(('Random Forest', RandomForestClassifier(n_estimators=10, \
        criterion='entropy', max_features=10, bootstrap=False)))
#    models.append(('Extra Trees', ExtraTreesClassifier(n_estimators=10, \
#        criterion='entropy', max_features=10, bootstrap=False)))

    # Train models 
    for model_name, model in models:
        if not args.quiet:
            print 'Training %s' % (model_name)
        model.fit(matrix, labels)

    # Save models and preprocessor 
    save_object_to_file(models, args.model_file)
    save_object_to_file(data_prep, args.preprocessor_file)

