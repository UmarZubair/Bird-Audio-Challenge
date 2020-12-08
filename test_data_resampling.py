import librosa
from scipy.io import wavfile
import numpy as np
import config

TOTAL_TEST_SAMPLES = 4512


def main():
    for i in range(TOTAL_TEST_SAMPLES):
        y = np.load(config.get_test_path()["RAW_DATA"] + str(i) + '.npy')
        _y = librosa.resample(y, 48000, 44100)
        wavfile.write(config.get_test_path()["DATA_PATH"] + "/" + str(i), 44100, _y)


if __name__ == "__main__":
    main()
