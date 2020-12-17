import numpy as np
import pandas as pd
import config
import feature_extract
import glob


def get_csv_data(csv):
    data_id = csv['itemid'].to_numpy()
    data_labels = csv['hasbird'].to_numpy()
    return data_id, data_labels


def train_data_to_numpy(data_set, data_path, data_id, data_labels):
    feature_file, label_file = feature_extract.feature_extract(data_path, data_id, data_labels)
    if data_set == 'ffbird':
        np.save('BAD_ff_feature', feature_file)
        np.save('BAD_ff_label', label_file)
    elif data_set == 'wwbird':
        np.save('BAD_ww_feature', feature_file)
        np.save('BAD_ww_label', label_file)


def test_data_to_numpy(data_path, data_id):
    feature_file = feature_extract.feature_extract(data_path, data_id, [])
    np.save('BAD_test_feature', feature_file)


def pseudo_data_to_numpy():
    x_test = []
    y_test = []
    test_data = np.load(config.get_test_path()["DATA_ARRAY_PATH"])
    csv_path = config.get_test_path()["CSV_FOR_PSEUDO_LABEL_PATH"]
    predicted_values = pd.read_csv(csv_path)['Predicted'].to_numpy()
    for i in range(4512):
        if predicted_values[i] > 0.93:
            x_test.append(test_data[i])
            y_test.append(1)
        if predicted_values[i] < 0.01:
            x_test.append(test_data[i])
            y_test.append(0)

    np.save('x_test.npy', x_test)
    np.save('y_test.npy', y_test)


def load_training_data():
    if not config.options()["DATA_IN_NUMPY_FORMAT"]:
        print('Saving ffbird dataset in numpy format')
        ff_data_path = config.get_training_path('ffbird')["DATA_PATH"]
        ff_csv = config.get_training_path('ffbird')["LABEL_PATH"]
        data_id, data_labels = get_csv_data(ff_csv)
        train_data_to_numpy('ffbird', ff_data_path, data_id, data_labels)

        print('Saving wwbird dataset in numpy format')
        ww_data_path = config.get_training_path('wwbird')["DATA_PATH"]
        ww_csv = config.get_training_path('wwbird')["LABEL_PATH"]
        data_id, data_labels = get_csv_data(ww_csv)
        train_data_to_numpy('wwbird', ww_data_path, data_id, data_labels)

    ff_array_path = config.get_training_path('ffbird')["DATA_ARRAY_PATH"]
    ww_array_path = config.get_training_path('wwbird')["DATA_ARRAY_PATH"]
    X_train = np.concatenate((np.load(ff_array_path), np.load(ww_array_path)), axis=0)
    y_train = np.concatenate((np.load(ff_array_path), np.load(ww_array_path)), axis=0)
    return X_train, y_train


def load_test_data():
    if not config.options()["DATA_IN_NUMPY_FORMAT"]:
        print('Saving test dataset in numpy format')
        data_path = config.get_test_path()["DATA_PATH"]
        data_id = np.arange(4512)
        train_data_to_numpy(data_path, data_id)

    test_array_path = config.get_test_path()["DATA_ARRAY_PATH"]
    X_test = np.load(test_array_path)
    return X_test


def load_pseudo_test_data():
    if not config.options()["PSEUDO_LABELLING"]:
        pseudo_data_to_numpy()
    x_pseudo_ = np.load(config.get_test_path()["PSEUDO_DATA_ARRAY_PATH"])
    y_pseudo_ = np.load(config.get_test_path()["PSEUDO_LABEL_ARRAY_PATH"])
    return x_pseudo_, y_pseudo_


def save_to_csv(results):
    submission_count = len(glob.glob('submissions/' + '*csv'))
    with open("submissions/submission_" + str(submission_count+1) + ".csv", "w") as fp:
        fp.write("ID,Predicted\n")
        for idx in range(4512):
            fp.write(f"{idx:05},{(results[idx][0])}\n")


