#!/bin/bash
project_dir="/Users/samway/Documents/Work/ML/projects/classification/"
scripts_dir=$project_dir
data_dir="/Users/samway/Documents/Work/ML/projects/classification/data/knights/"
test_dir="/Users/samway/Documents/Work/ML/projects/classification/tests/"

cd $project_dir
for study in $(ls $data_dir)
do
    echo "Processing $study..."
    $scripts_dir/evaluate_models.py -i $data_dir/$study/otu.biom -l $data_dir/$study/labels.txt -o $test_dir/$study.txt -k 6
done
echo "Done"

