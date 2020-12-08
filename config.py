def get_train_path(data_set):
    if data_set == 'ffbird':
        paths = {
            "DATA_PATH": "data/ffbird/waves",  # Path where the data is in waves for dataset 1
            "LABEL_PATH": "data/ffbird/ff1010bird_metadata_2018.csv",  # Path for the labels for dataset 1
            "DATA_ARRAY_PATH": "data/ffbird/np/BAD_ff_feature.npy",  # Path where the data in numpy format are saved
            "LABEL_ARRAY_PATH": "data/ffbird/np/BAD_ff_label.npy"  # Path where the labels in numpy format are saved
        }
    elif data_set == 'wwbird':
        paths = {
            "DATA_PATH": "data/wwbird/waves",
            "LABEL_PATH": "data/wwbird/warblrb10k_public_metadata_2018.csv",
            "DATA_ARRAY_PATH": "data/wwbird/np/BAD_wblr_feature.npy",
            "LABEL_ARRAY_PATH": "data/wwbird/np/BAD_wblr_label.npy"
        }
    return paths


def get_test_path():
    paths = {
        "DATA_PATH": "data/test_data/waves",  # Path where the data is in waves. Run test_data_resampling.py first if not
        "DATA_ARRAY_PATH": "data/test_data/np/BAD_test_feature.npy",  # Path where the data in numpy format is saved
        # Best submission csv file which is being used to pseudo labeling
        "CSV_FOR_PSEUDO_LABEL_PATH": "submissions/submission_for_pseudo_label/submission_57.csv",
        "PSEUDO_DATA_ARRAY_PATH": "data/test_data/pseudo_data/x_test.npy",  # Pseudo labelled train data
        "PSEUDO_LABEL_ARRAY_PATH": "data/test_data/pseudo_data/y_test.npy",  # Pseudo labelled label data
        "RAW_DATA": "data/test_data/raw/", # This is the test data in numpy format as downloaded from kaggle competition
        "BEST_SUBMISSIONS" : "submissions/best_submissions/" # Where the best submissions are for ensembling
    }
    return paths


def options():
    """
    For first time run.
    Set the below options to False to make the data in numpy arrays for ease in later uses.
    Afterwards change these back to True.
    """
    options = {
        "DATA_IN_NUMPY_FORMAT": True,  # Set this to false if the data is not yet in numpy format.
        "PSEUDO_LABELLING": True  # Set this to false if the pseudo labelled data in not in numpy format
    }
    return options
