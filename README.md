# Bird Audio Detection Kaggle Challenge
 1st place solution for the InClass Kaggle competition of the DATA.ML.200-2020-2021-1 Pattern Recognition and Machine Learning course:
 https://www.kaggle.com/c/bird-audio-detection
 
## Running the code:
Training and test datasets can be downloaded from https://www.kaggle.com/c/bird-audio-detection/overview/datasets. <br/>
Place the data in format:

    .
    ├── data                                          
    │   ├── ffbird                                    # Training set freefield1010
    │   │   ├── waves                                 # Audio files
    │   │   └── ff1010bird_metadata_2018.csv          # Labels
    │   ├── wwbird                                    # Training set warblrb10k
    │   │   ├── waves                                 # Audio files for warblrb10k
    │   │   └── warblrb10k_public_metadata_2018.csv   # Labels
    │   ├── test_data                                 # Test set
    │   │   ├── waves                                 # Audio files
    │   │   ├── pseudo_data                           # Pseudo data path 
    |   |   └── raw                                   # This is the test data in numpy format as downloaded from kaggle competition
    └── submissions
    │   ├── best_submissions                          # Path to best submissions for ensembling
    |   └── submission_for_pseudo_label               # Path to best submission csv file which is being used to pseudo labeling
    └── README.md
    
Insure paths in config.py and for first time run test_data_resampling.py. After that run any of the model files for direct training.

## Project approach:

### Pre-processing:
So, our feature extraction was based on this research paper:
https://www.researchgate.net/profile/Daksh_Thapar/publication/327388806_All-Conv_Net_for_Bird_Activity_Detection_Significance_of_Learned_Pooling/links/5d42aa3992851cd0469700a1/All-Conv-Net-for-Bird-Activity-Detection-Significance-of-Learned-Pooling.pdf

The audio signal is windowed using a frame size of 20 ms with no overlap and a Hamming window. 882 FFT points are used to obtain the Fourier transform of an audio frame. This process converts a 10 second audio recording into a 441 × 500-dimensional spectrogram representation. Each frame of this spectrogram is converted into a 40-dimensional vector of log filter bank energies using Mel filter-bank. Hence, each 10 second audio was represented in 40 × 500 dimensional Mel spectrogram.

Random Mel spectrograms from training set (ffbird and wwbird datasets):
 
![alt text](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2246613%2F4e825a166a674cc2c3fa2bf3651831ec%2Fdownload%20(2).png?generation=1607441071778046&alt=media)<br/>
![alt-text](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2246613%2F60a9b6b65d3217854028d0dc0781fa54%2Fdownload%20(4).png?generation=1607441133677808&alt=media)<br/>
![alt-text](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2246613%2F5174d4765658f908fc5f975340023dd4%2Fdownload%20(5).png?generation=1607441173462790&alt=media)<br/>

Random Mel spectrograms from test set:
![alt-text](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2246613%2Fab962a57f507bdee0e73616df10454e6%2Fdownload%20(7).png?generation=1607441358018610&alt=media)<br/>
![alt-text](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2246613%2Fc4a985d1c7d6aaf6e48d3e392e7ad368%2Fdownload%20(8).png?generation=1607441365106166&alt=media)<br/>

Test data was resampled to 44100 and saved as waves where the same pre-processing was done on it. In the test dataset, one challenge was the high noise of the audio files. To resolve this, we tried to denoise the datasets and save the Mel spectrograms as images. This was done using the noisereduce library by Timsaimb.

Random denoised images with bird sound present:

![alt-text](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2246613%2F7c23876a2e7e61a5be7a0bd1848aa384%2F0.png?generation=1607441480080141&alt=media)<br/>
![alt-text](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2246613%2F5bd4a7065ea2aece8ab14fadab460d20%2F1.png?generation=1607441496590404&alt=media)<br/>

Denoised images with no bird sound:

![alt-text](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2246613%2F0cd7b6c55dcc3a364bd5bc0a1eff61e0%2F21.png?generation=1607441651312789&alt=media)<br/>
![alt-text](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2246613%2F3da0b89f2572b16ce46f3e3114e4531a%2F5.png?generation=1607441719086068&alt=media)<br/>

For us, this looked promising in the start, yet in the end we were not able to get good accuracy on the denoised data:
68 percent using denoised
72 percent without denoising

### Models:
We went over different models, I will include 3 of those here on which we spent the most time on.

First model was a covnet model which is mentioned in the above paper.

In comparison with the one described in paper, we changed the filters to 40 instead of 16, added another dense layer of 196, changed the output of last layer to 1 and changed it to sigmoid instead of softmax. Binary-crossentropy was the loss function and Adam was used as the optimizer. Dropout of 0.25 and 0.50 was used after every layer. Using around 200 epochs with early stopping, it took around 25-30 minutes for 80-20 split training and we were able to get 69-70 percent accuracy in the public leaderboard on this model.

The second model was a CRNN approach whose inspiration was the Kostas research paper and the Kaggle notebook mentioned below. 4 bidirectional with CuDNNLSTM layers of 128 filters were implemented. With relu as activation function, Adam optimizer and binary cross entropy. Attention layer of 500 filter was implemented before the last dense layer which was taken from this Kaggle notebook:
https://www.kaggle.com/kbhartiya83/sound-recogniser-bidirectionallstm-attention
The maximum accuracy we reached from the CRNN approach was 68.

For the third model, we went back in search for CNN models and tried different models like Alexnet and LeNET. Yet a simple CNN model in the end turned out to perform the best. 3 layers of 48 filter size followed by 3 layers of 96 filter. With 2 dense layers of 384 and 1. Kernel size 3 in the first two and 5 in the third for both 48 and 96 layers. ReLU was used as the activation function and with data augmentation of shift range, height range and horizontal flip we were able to reach 72 accuracy on the public leaderboard. 

### Training:
We used the pseudo label-approach where we included 1500 samples from test data which had either more than 0.95 or less than 0.10 on the best submission. One thing we noticed was that both the samples of bird present and bird not present needed to be near equal. Otherwise, if for instance we only took 1000 test samples where all the samples had bird present and added it in training. Then the model was always resulting in 3-4 percent drop in accuracy.

### Ensembling:
For ensembling, I used a simple approach to take the mean of either the high threshold values or the low threshold values. 
Results:

Best approach without ensembling:<br/>
Public : 0.72406 Private : 0.70369<br/>
Best 3 submissions ensembling:<br/>
Public : 0.72619 Private : 0.70726<br/>
Best 11 submissions ensembling:<br/>
Public : 0.72926 Private : 0.70497<br/>

