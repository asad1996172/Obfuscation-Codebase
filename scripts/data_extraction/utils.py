"""
Contains all utility functions for data extraction and manipulation
"""

from os import walk
import io
import numpy as np
import pickle
from operator import itemgetter
import os
from sklearn import preprocessing
from prettytable import PrettyTable


def get_authors(dataset_path, authors_required):
    """
    This function gets the names of authors with highest average
    number of characters per document, that are within the input range

    :param dataset_path: path of required dataset
    :param authors_required: tuple containing range e.g., (0,5)

    :return: list of author names in the given range
    """

    # Read all author names in the given dataset
    authors_names = []
    for (_, directories, _) in walk(dataset_path):
        authors_names.extend(directories)
        break

    # Create a dictionary containing authors and average number of characters in their articles
    author_name_with_number_of_characters = {}
    for author in authors_names:
        # Read all articles by a given author
        author_files = []
        for (_, _, filenames) in walk(dataset_path + "/" + str(author)):
            author_files.extend(filenames)
            break
        author_files = sorted(author_files)
        # Calculate number of characters per article
        no_of_characters_per_article = []
        for i in range(len(author_files)):
            filename = author_files[i]
            file_path = dataset_path + "/" + str(author) + "/" + str(filename)
            # Read file text
            input_text = io.open(file_path, "r", errors="ignore").readlines()
            input_text = ''.join(str(e) + "" for e in input_text)

            no_of_characters_per_article.append(len(input_text))

        author_name_with_number_of_characters[author] = np.mean(no_of_characters_per_article)

    # Using pre-set top 10 authors for ebg and blogs
    ebg_10 = ['h', 'qq', 'pp', 'm', 'y', 'ss', 'w', 'b', 'c', 'o']
    blogs_10 = ['1151815', '554681', '2587254', '3010250', '3040702', '3403444', '215223', '1046946', '1476840',
                '1234212']

    sorted_authors = (sorted(
        author_name_with_number_of_characters.items(), key=itemgetter(1), reverse=True
    ))
    sorted_authors = [author_name for author_name, _ in sorted_authors]

    if 'ebg' in dataset_path:
        sorted_authors = ebg_10 + [author for author in sorted_authors if author not in ebg_10]
    elif 'blogs' in dataset_path:
        sorted_authors = blogs_10 + [author for author in sorted_authors if author not in blogs_10]

    return sorted_authors[authors_required[0]:authors_required[1]]


def prepare_data_sets_for_attribution_experiments(dataset_path, authors_list, pickled_dataset_path, dataset_name):
    """
    This function prepares dataset for authorship attribution experiments

    :param dataset_path: path of required dataset
    :param authors_list: list of authors to be used in experiments
    :param pickled_dataset_path: path to save dataset at
    :param dataset_name: name of dataset
    """

    if not os.path.exists(pickled_dataset_path):
        os.makedirs(pickled_dataset_path)

    le = preprocessing.LabelEncoder()
    le.fit_transform(authors_list)
    np.save(pickled_dataset_path + '/classes.npy', le.classes_)

    x_train = []
    x_test = []
    for author in authors_list:
        author_files = []
        for (_, _, filenames) in os.walk(dataset_path + "/" + str(author)):
            author_files.extend(filenames)
            break

        # important to sort for consistency
        author_files = sorted(author_files)
        count = 0
        for i in range(len(author_files)):
            filename = author_files[i]
            file_path = dataset_path + "/" + str(author) + "/" + str(filename)
            input_text = io.open(file_path, "r", errors="ignore").readlines()
            input_text = ''.join(str(e) + "" for e in input_text)
            count += 1
            author_id = le.transform([author])[0]
            if dataset_name == 'ebg':
                if count <= 12:
                    x_train.append((file_path, filename, author_id, author, input_text))
                else:
                    x_test.append((file_path, filename, author_id, author, input_text))
            elif dataset_name == 'blogs':
                if count <= 80:
                    x_train.append((file_path, filename, author_id, author, input_text))
                else:
                    x_test.append((file_path, filename, author_id, author, input_text))
    # Store data (serialize)
    with open(pickled_dataset_path + '/X_train.pickle', 'wb') as handle:
        pickle.dump(x_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(pickled_dataset_path + '/X_test.pickle', 'wb') as handle:
        pickle.dump(x_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Total train documents : ', len(x_train))
    print('Total test documents : ', len(x_test))


def print_dataset_stats(dataset_path):
    """
    This function is for printing out stats of a created dataset. It
    shows the number of train and test documents from each author

    :param dataset_path: path of dataset
    """

    x_train = pickle.load(open(dataset_path + '/X_train.pickle', 'rb'))
    x_test = pickle.load(open(dataset_path + '/X_test.pickle', 'rb'))

    train_data = []
    for file_path, filename, author_id, author, input_text in x_train:
        train_data.append(author)

    test_data = []
    for file_path, filename, author_id, author, input_text in x_test:
        test_data.append(author)

    train_data = dict(zip(list(train_data), [list(train_data).count(i) for i in list(train_data)]))
    test_data = dict(zip(list(test_data), [list(test_data).count(i) for i in list(test_data)]))

    x = PrettyTable()
    x.title = dataset_path.split('/')[-1].upper()
    x.field_names = ['Author', 'Train', 'Test']
    for author in sorted(train_data.keys()):
        x.add_row([author, train_data[author], test_data[author]])
    x.add_row(['', '', ''])
    x.add_row(['Totals', len(x_train), len(x_test)])
    print("\n", x, "\n")


if __name__ == "__main__":
    # Get ebg top 5 authors
    ebg_authors = get_authors(dataset_path="../../data_files/datasets/ebg", authors_required=(0, 5))
    print("Author Names: ", ebg_authors)
    prepare_data_sets_for_attribution_experiments(
        dataset_path="../../data_files/datasets/ebg",
        authors_list=ebg_authors,
        pickled_dataset_path='../../data_files/datasets/prepared_datasets/' + 'ebg' + '/experiment_0_5/',
        dataset_name="ebg"
    )
    print_dataset_stats("../..//data_files/datasets/prepared_datasets/ebg/experiment_0_5")
