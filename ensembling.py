import numpy as np
import glob
import config
import utils
import pandas as pd

TOTAL_TEST_SAMPLES = 4512


def emsemble():
    submissions = glob.glob(
        config.get_test_path()["BEST_SUBMISSIONS"] + "*.csv")  # These should be odd number of best submissions.
    best_sub = []

    for submission in submissions:
        best_sub.append(pd.read_csv(submission)['Predicted'].to_numpy())

    best_sub = np.array(best_sub)
    best_results = []

    for id in range(TOTAL_TEST_SAMPLES):
        high_threshold = []
        low_threshold = []
        for sub in range(len(submissions)):
            if best_sub[sub][id] > 0.5:
                high_threshold.append(best_sub[sub][id])
            else:
                low_threshold.append(best_sub[sub][id])

        if len(high_threshold) > len(low_threshold):
            best_results.append(np.mean(high_threshold, axis=0))
        else:
            best_results.append(np.mean(low_threshold, axis=0))

    best_results = np.array(best_results)
    utils.save_to_csv(best_results)


emsemble()
