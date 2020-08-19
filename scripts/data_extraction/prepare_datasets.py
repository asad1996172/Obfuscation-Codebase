"""
This module is for creating pickle datasets that are ready
for authorship attribution experiments.

Example
---------
To create ebg dataset pickles for authors 0 to 5, run the following
    python3 prepare_datasets.py \
        --dataset-path ../../data_files/datasets/ebg \
        --required-range 0 5
"""

import argparse
import utils

def data_preparation(dataset_path, required_range):
    """
    This function gets the names of authors with highest average
    number of characters per document, that are within the input range
    and then creates their pickled dataset for authorship attribution
    experiments

    :param dataset_path: path of required dataset
    :param authors_required: tuple containing range e.g., (0,5)
    """

    ebg_authors = utils.get_authors(
        dataset_path=dataset_path, authors_required=required_range)
    print("Author Names: ", ebg_authors)
    utils.prepare_data_sets_for_attribution_experiments(
        dataset_path=dataset_path,
        authors_list=ebg_authors,
        pickled_dataset_path='../../data_files/datasets/prepared_datasets/{}/experiment_{}_{}/'.format(
            dataset_path.split('/')[-1],
            required_range[0],
            required_range[1]    
        ),
        dataset_name=dataset_path.split('/')[-1]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset-path', required=True, type=str)
    parser.add_argument('-rr', '--required-range', required=True, nargs='+', type=int)
    args = parser.parse_args()

    data_preparation(
        args.dataset_path,
        tuple(args.required_range)
    )

