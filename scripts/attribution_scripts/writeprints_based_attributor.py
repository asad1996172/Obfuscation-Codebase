"""
This module is for training a machine learning classifier
with stylometric feature set. 

Usage
------
python3 writeprints_based_attributor.py \
    --dataset-path ../../data_files/datasets/prepared_datasets/ebg/experiment_0_5 \
    --trained-model-path ../../data_files/attribution_models/ebg/experiment_0_5/ \
    --feature-set writeprints-static \
    --classifier-name rfc \
    --corpus-path ../../data_files/datasets/ebg
"""
import argparse
import pickle
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

sys.path.append('../feature_extraction/writeprints_features')
import writeprintsStatic as ws
from JstyloFeatureExtractor import jstyloFeaturesExtractor


def get_data(dataset_path, corpus_path, feature_set):
    """
    To read authors and their text from pickled data in test and train sets

    Parameters
    ----------
    dataset_path: str
        path at which pickle data is placed
    corpus_path: str
        path of corpus for creating features
    feature_set: str
        name of stylometric feature set to use
    
    Return
    ----------
    data: tuple
        tuple containing x_train, x_test, y_train and y_test
    """

    jfe = jstyloFeaturesExtractor(corpus_path)

    with open('{}/X_train.pickle'.format(dataset_path), 'rb') as handle:
        all_train = pickle.load(handle)

    with open('{}/X_test.pickle'.format(dataset_path), 'rb') as handle:
        all_test = pickle.load(handle)

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    # data format in pickled file => (file_path, filename, author_id, author, input_text)

    for (_, _, author_id, _, input_text) in all_train:
        if feature_set == "writeprints-static":
            x_train.append(ws.calculateFeatures(input_text, jfe))
        y_train.append(author_id)
    
    for (_, _, author_id, _, input_text) in all_test:
        if feature_set == "writeprints-static":
            x_test.append(ws.calculateFeatures(input_text, jfe))
        y_test.append(author_id)

    return x_train, x_test, y_train, y_test


def train_classifier(trained_model_path, classifier_name, data, feature_set):
    """
    This function trains the attribution classifier

    Parameters
    -----------
    trained_model_path: str
        path at which the final trained model will be saved at
    
    classifier_name: str
        name of classifier that is to be used for training (rfc, svm)
    
    data: tuple
        tuple containing x_train, x_test, y_train and y_test
    
    feature_set: str
        name of stylometric feature set to use

    """
    x_train, x_test, y_train, y_test = data
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    if classifier_name == 'rfc':
        clf = RandomForestClassifier(n_estimators=50)
    elif classifier_name == 'svm':
        clf = SVC(kernel='linear', random_state=10)
    
    print("Starting Training...")
    clf.fit(x_train, y_train)
    print("Starting Saving...")

    trained_model_path = '{}/{}_{}'.format(
        trained_model_path, classifier_name, feature_set)
    if not os.path.exists(trained_model_path):
        os.makedirs(trained_model_path)

    filename = '{}/trained_model.sav'.format(trained_model_path)
    pickle.dump(clf, open(filename, 'wb'))
    pickle.dump(scaler, open('{}/std_scaler.pkl'.format(trained_model_path), 'wb'))
    
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    scalar = pickle.load(
        open('{}/std_scaler.pkl'.format(trained_model_path), 'rb'))
    x_test = scaler.transform(x_test)

    predicted = loaded_model.predict(x_test)
    print("Test Accuracy : ", accuracy_score(y_test, predicted))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset-path', required=True, type=str)
    parser.add_argument('-tmp', '--trained-model-path',required=True, type=str)
    parser.add_argument('-fs', '--feature-set',
                        required=True, type=str)
    parser.add_argument('-cn', '--classifier-name',
                        required=True, type=str)
    parser.add_argument('-cp', '--corpus-path',
                        required=True, type=str)
    args = parser.parse_args()

    os.environ["WRITEPRINTS_RESOURCES"] = "../feature_extraction/writeprints_features/writeprintresources"
    
    data = get_data(args.dataset_path, args.corpus_path, args.feature_set)
    train_classifier(args.trained_model_path, args.classifier_name, data, args.feature_set)
