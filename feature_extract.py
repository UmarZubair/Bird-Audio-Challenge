import librosa
import scipy.signal
import numpy as np
import os

N_MELS = 40
FFT_POINTS = 882 * 2
SR = 44100
HAMMING_SIGNAL = scipy.signal.hamming
FRAMES = 500


def feature_extract(path, data_id, data_labels):
    feature_file = []
    label_file = []

    for i in range(len(data_id)):
        [wave, _] = librosa.core.load(os.path.join(path, str(data_id[i])) + ".wav",
                                      sr=SR)  # read wav file (fs = 44.1 kHz)
        wave = librosa.stft(wave, FFT_POINTS, FFT_POINTS,
                            HAMMING_SIGNAL)  # STFT computation (fft_points = 882*2, overlap= 50%, analysis_window=40ms)
        wave = np.abs(wave) ** 2
        spectrogram = librosa.feature.melspectrogram(S=wave, n_mels=N_MELS)  # mel bands (40)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        norm_spectrogram = spectrogram - np.amin(spectrogram)
        norm_spectrogram = norm_spectrogram / float(np.amax(norm_spectrogram))

        if int(norm_spectrogram.shape[1]) < FRAMES:  # 10 sec samples gives 500 frames
            z_pad = np.zeros((N_MELS, FRAMES))
            z_pad[:, :-(FRAMES - norm_spectrogram.shape[1])] = norm_spectrogram
            feature_file.append(z_pad)
        else:
            img = norm_spectrogram[:, np.r_[0:FRAMES]]  # final_shape = 40*500
            feature_file.append(img)

        if len(data_labels) > 0:
            label_file.append(data_labels[i])

    feature_file = np.array(feature_file)
    feature_file = np.reshape(feature_file, (len(data_id), N_MELS, FRAMES, 1))

    if len(data_labels) > 0:  # In case of training data
        label_file = np.array(label_file)
        return feature_file, label_file
    else:  # In case of testing data
        return feature_file
