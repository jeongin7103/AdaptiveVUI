import os
import sys
import csv
import numpy as np
from scipy import signal
import pandas as pd

from mne.preprocessing import ICA
import mne
import torch
import time
import datetime
import braindecode

##############################################################################
##############################################################################
##############################################################################

def notch_filter(x, l_cut, h_cut, fs, filt_order):
    import warnings
    warnings.simplefilter('ignore')
    nyq_freq = 0.5 * fs
    low = l_cut / nyq_freq
    high = h_cut / nyq_freq
    b, a = signal.butter(filt_order, [low, high], btype="bandstop")
    y = signal.filtfilt(b, a, x)
    return y

def low_pass_filter(x, high_cut, fs, filt_order):
    import warnings
    warnings.simplefilter('ignore')
    nyq_freq = 0.5 * fs
    low = high_cut / nyq_freq
    b, a = signal.butter(filt_order, low, btype="lowpass")
    y = signal.filtfilt(b, a, x)
    return y

def high_pass_filter(x, low_cut, fs, filt_order):
    import warnings
    warnings.simplefilter('ignore')
    nyq_freq = 0.5 * fs
    high = low_cut / nyq_freq
    b, a = signal.butter(filt_order, high, btype="highpass")
    y = signal.filtfilt(b, a, x)
    return y

def remove_fp(x):
    eog = x[:,(0, 7)]
    x = x.transpose()
    x = np.delete(x, (0,7), axis=0)
    x = x.transpose()
    x = np.append(x, eog, axis=1)
    return x

def eog_artifact_remove(x, s_rate, n_chan, n_eog):
    # Temporal 10 ch
    ch_names = ['FT9', 'T7', 'TP9', 'TP10', 'T8', 'FT10', 'FT7', 'C5', 'TP7', 'TP8', 'C6', 'FT8', 'EOG1', 'EOG2']
    info = mne.create_info(ch_names, s_rate, ch_types=["eeg"] * n_chan + ["eog"] * n_eog)
    raw = mne.io.RawArray(x.T, info)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    raw_tmp = raw.copy()
    ica = mne.preprocessing.ICA(method="infomax",
                                fit_params={"extended":True},
                                random_state=1)
    ica.fit(raw_tmp)
    ica.exclude = []
    eog_indices, eog_scores = ica.find_bads_eog(raw_tmp)
    ica.exclude = eog_indices
    raw_correct = raw.copy()
    ica.apply(raw_correct)
    x = raw_correct._data
    x = np.delete(x, (10,11), axis=0)
    x = x.transpose()
    return x

def Preprocessing(x, low_cut, high_cut, fs, target_fs):
    # 0. Channel remove
    x = remove_fp(x)
    x = x.transpose()
    x = x.transpose()
    for i in range(0,len(x)):
        # 1. Notch filtering
        x[i] = notch_filter(x[i], 59, 61, fs, filt_order=2)
        # 2. Lowpass filtering
        x[i] = low_pass_filter(x[i], high_cut, fs, filt_order=2)
        # 3. Highpass filtering
        x[i] = high_pass_filter(x[i], low_cut, fs, filt_order=2)
    # 4. Data reconstruction
    x = x.transpose()
    # 5. Down-sampling (1,000 Hz -> 200 Hz)
    x = signal.resample(x, round(len(x)*(target_fs/fs)), domain='time')
    # 6. ICA
    x = eog_artifact_remove(x, s_rate = 200, n_chan = 12, n_eog = 2)
    # 8. Up-sampling (n Hz -> 1,600 Hz)
    x = signal.resample(x, round(len(x)*(1600/len(x))), domain='time').transpose()
    
    return x

##############################################################################
##############################################################################
##############################################################################
def eeg_predict(record_start_time, ans_start_time, ans_end_time):
    # Data load
    np.set_printoptions(threshold=sys.maxsize)
    print(torch.cuda.is_available())
    ans_start = datetime.datetime.strptime(ans_start_time, "%H:%M:%S.%f") - datetime.datetime.strptime(record_start_time, "%H:%M:%S.%f")
    ans_end = datetime.datetime.strptime(ans_end_time, "%H:%M:%S.%f") - datetime.datetime.strptime(record_start_time, "%H:%M:%S.%f")
    timestamp = [ans_start.seconds*1000 + ans_start.microseconds/1000,ans_end.seconds*1000 + ans_end.microseconds/1000]
    path = '/home/fcsl/Desktop/VUI experiment'     # Data path
    os.chdir(path)
    eeg = []


    state = True
    print("STATE: ", state)

    while state:
        eeglength = open("Data1s.csv")
        lengthRead = csv.reader(eeglength)
        lengthNew = len(list(lengthRead))
        if timestamp[1] > lengthNew:
            print("Control False: ","| Timestamp: ", timestamp[1],"Lenght: ", lengthNew)
            state = True

        else:
            # Loading PART

            print("Control True: ","| Timestamp: ", timestamp[1],"Lenght: ", lengthNew)
            #time.sleep(1)
            if os.path.isfile("Data1s.csv") and os.access("Data1s.csv", os.R_OK): # checking
                with open("Data1s.csv") as file_name:  # Data file(.csv) name
                    eeg = np.loadtxt(file_name, delimiter=",").astype(np.float32) # anwer to our probs
            eeg = eeg.transpose()
            state=False
            #state = False



    print(timestamp)
    print("Crop start column: ", int(timestamp[0]))
    print("Crop end column: ", int(timestamp[1]))


    try:
        time.sleep(1)
        print("Start TRY")
        print("Control 2 True: ", "| Timestamp: ", timestamp[1], "Lenght: ", lengthNew)
        #print("Indices probs",eeg)
        eeg = eeg[:,int(timestamp[0]):int(timestamp[1])]
        #print("EEG Cropped Array: ", eeg)

        # Data preprocessing
        try:
            eeg = Preprocessing(eeg, low_cut = 1, high_cut = 80,
                                fs = 1000, target_fs = 200)
        except:
            pass
        print("End TRY")
    except:
        print("Start except")
        # time.sleep(1)
        # print("Control 3 True: ", "| Timestamp: ", timestamp[1], "Lenght: ", lengthNew)
        # if os.path.isfile("Data1s.csv") and os.access("Data1s.csv", os.R_OK):
        #     with open("Data1s.csv") as file_name:  # Data file(.csv) name
        #         eeg = np.loadtxt(file_name, delimiter=",").astype(np.float32)
        # eeg = eeg.transpose()
        while state:
            eeglength = open("Data1s.csv")
            lengthRead = csv.reader(eeglength)
            lengthNew = len(list(lengthRead))
            if timestamp[1] > lengthNew:
                print("Control False: ", "| Timestamp: ", timestamp[1], "Lenght: ", lengthNew)
                state = True

            else:
                print("Control True: ", "| Timestamp: ", timestamp[1], "Lenght: ", lengthNew)
                # time.sleep(1)
                if os.path.isfile("Data1s.csv") and os.access("Data1s.csv", os.R_OK):
                    with open("Data1s.csv") as file_name:  # Data file(.csv) name
                        eeg = np.loadtxt(file_name, delimiter=",").astype(np.float32)  # anwer to our probs
                eeg = eeg.transpose()
                state = False

        eeg = eeg[:, int(timestamp[0]):int(timestamp[1])]


        # Data preprocessing
        eeg = Preprocessing(eeg, low_cut=1, high_cut=80,
                            fs=1000, target_fs=200)
        print("END Except")

    
    ########## ShallowNet classification ##########
    # Training된 model 불러오기
    os.chdir('/home/fcsl/Desktop/VUI experiment/Demo')        # Model path
    #from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
    #model = ShallowFBCSPNet(in_chans=14, n_classes=2, final_conv_length=20)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = torch.load('Trained_model_LYB.pt', map_location=torch.device('cuda:0'))  # Model name
    model = model
    X_test = eeg
    X_test = np.reshape(X_test,(1,len(eeg),len(eeg[0])))            # ShallowNet의 Input으로 사용하기 위한 차원 변경 (2D -> 3D)
    
    # Evaluation
    from sklearn.metrics import confusion_matrix
    y_out = model.predict_outs(X_test)
    y_out = np.exp(y_out)
    return y_out
