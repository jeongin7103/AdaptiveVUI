import os
import time
import keyboard
import matplotlib.pyplot as plt
import speech_recognition as sr
from playsound import playsound
import random
import multiprocessing
import cv2
import numpy as np
from danMtlTrt.inference50_t import main, calctime,inference
from IPython.display import display, clear_output
from pynput.keyboard import Key, Listener
from RDALatest_partitioned_array import mainRDAEEG
from BRL_EEG_python_timestamp_PA import eeg_predict
from datetime import datetime

from pynput import keyboard

TimeStamps = []
TimeStampsEEG = [datetime.strftime(datetime.now(),'%H:%M:%S.%f'),datetime.strftime(datetime.now(),'%H:%M:%S.%f')]
triggerGlobal = False

manager = multiprocessing.Manager()
return_dict = manager.dict()


model1 = inference(4, "/home/fcsl/Desktop/VUI experiment/danMtlTrt/models/resnet50_ft_weight.pkl")  # model2 path


FERval = [float(),float()]
FERvalFinal = float()
EEGvalFinal = float()
EEGval = []


def getFERval():
    global FERval
    global FERvalFinal
    FERvalFinal = FERval[1]
    # if FERval[0] > FERval[1]:
    #     FERvalFinal = FERval[0]
    # else:
    #     FERvalFinal = FERval[1]
    return FERvalFinal


def getEEGval():
    global EEGval
    global EEGvalFinal
    EEGvalFinal = EEGval[0][1]
    #print("EEG VALUE: ",EEGval)
    # if EEGval[0][0] > EEGval[0][1]:
    #     EEGvalFinal = EEGval[0][0]
    # else:
    #     EEGvalFinal = EEGval[0][1]
    return EEGvalFinal


def emotionPlotting():
    global curQudrant
    global fig
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.title('<Inference Engine>')
    plt.xlabel("Trustability (EEG)")
    plt.ylabel("Stability (FER)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.axvline(x=0.5, c='black')
    plt.axhline(y=0.5, c='black')
    ax.set_xlim([0.0, 1])
    ax.set_ylim([0.0, 1])

    # Get FER & EEG value
    FER = float(getFERval())
    EEG = float(getEEGval())
    #print("EEG 만족도: ", EEG, "FER 만족도: ", FER)

    if EEG > 0.5 and FER > 0.5:
        print("Q1")
        rectangle = plt.Rectangle((0.5, 0.5), 0.5, 0.5, fc='moccasin', ec="black")
        ax.add_patch(rectangle)
        plt.scatter(EEG, FER, s=12 ** 2, c='seagreen', edgecolors='black')
        plt.text(
            EEG, FER + 0.03,
                    "(%.3f,%.3f)" % (EEG, FER),
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=12
        )

    elif EEG < 0.5 and FER > 0.5:
        print("Q2")
        rectangle = plt.Rectangle((0.0, 0.5), 0.5, 0.5, fc='moccasin', ec="black")
        ax.add_patch(rectangle)
        plt.scatter(EEG, FER, s=12 ** 2, c='seagreen', edgecolors='black')
        plt.text(
            EEG, FER + 0.03,
                 "(%.2f,%.2f)" % (EEG, FER),
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=12
        )
    elif EEG < 0.5 and FER < 0.5:
        print("Q3")
        rectangle = plt.Rectangle((0.0, 0.0), 0.5, 0.5, fc='moccasin', ec="black")
        ax.add_patch(rectangle)
        plt.scatter(EEG, FER, s=12 ** 2, c='seagreen', edgecolors='black')
        plt.text(
            EEG, FER + 0.03,
                 "(%.2f,%.2f)" % (EEG, FER),
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=12
        )
    elif EEG > 0.5 and FER < 0.5:
        print("Q4")
        rectangle = plt.Rectangle((0.5, 0.0), 0.5, 0.5, fc='moccasin', ec="black")
        ax.add_patch(rectangle)
        plt.scatter(EEG, FER, s=12 ** 2, c='seagreen', edgecolors='black')
        plt.text(
            EEG, FER + 0.03,
                 "(%.2f,%.2f)" % (EEG, FER),
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=12
        )
    # plotting

    ax.plot()
    display(fig)
    plt.pause(1)
    #clear_output(wait=True)

def spaceTrigger(key):#key):



    if key == keyboard.Key.space:
        # Stop listener
        print("Starting Inference")
        #KWS()
        try:

            KWS()
        except:
            os.system("python trading.py")


if key == keyboard.Key.esc:
        # Stop listener
        return False
    pass



def inputVoice(lang):
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        playsound('/home/fcsl/Desktop/VUI experiment/beep.mp3')
        print("Go on Speak")
        r.pause_threshold = 1
        time.sleep(1)
        audio = r.listen(source)

        try:
            Question = [r.recognize_google(audio, language=lang)] #ko~en
            print("Your Question is: ", Question)
        except Exception as e:
            print("Error ", e)


    return Question[0].split()

def responseGenerator():
    #global responseArray
    responseArray = ["MS", "MR", "WR", "WS"]
    respo = random.choice(responseArray)
    return respo

def txtkeywordList():
    with open('/home/fcsl/Desktop/VUI experiment/kwlist.txt') as f:
        kl = f.read().splitlines()
        #print(kl)
    return kl

def timeStampRec(startTime, EndTime):
    timeLog = open("/home/fcsl/Desktop/VUI experiment/timestamp.txt", "w")
    timeLog.write(startTime, EndTime)
    pass

def KWS():
    #plt.close() # Reinitialize plot
    global TimeStamps
    global triggerGlobal
    global TimeStampsEEG
    global EEGval
    EEG_array = []

    userQuestion = inputVoice("ko")
    keyword_list = txtkeywordList()
    superKeyword = set(keyword_list) & set(userQuestion)

    if superKeyword == set():

        playsound('/home/fcsl/Desktop/VUI experiment/null.mp3')
        return
    else:
        supkey = list(superKeyword)
        if supkey[0] in userQuestion:
            vuiResponse = responseGenerator()

            for kw in txtkeywordList():
                if supkey[0] == kw:
                    path = '/home/fcsl/Desktop/VUI experiment/respo/' + vuiResponse + "/" + supkey[0] + '.mp3'

                    startTime = time.strftime('%H:%M:%S', time.localtime(time.time()))
                    EEG_array = mainRDAEEG(path)

                    #print("EEG Array from new RDA: ", len(EEG_array))

                    eeg_result = eeg_predict(EEG_array)
                    print("EEG result: ", eeg_result)
                    EEGval = eeg_result

                    endTime = time.strftime('%H:%M:%S', time.localtime(time.time()))

                    TimeStamps = [startTime, endTime]
                    FERinfer()
                    emotionPlotting()
                    time.sleep(1)

def FERinfer():
    global model1
    global return_dict
    global manager
    global TimeStamps
    global FERval

    li = calctime(TimeStamps[0], TimeStamps[1])
    start = time.time()
    #print(return_dict)
    ndic = return_dict.copy()

    result = np.array([[0.0, 0.0]])

    i = li[-1]

    if i in ndic:

        result[0] = model1.pred(ndic[i])
        FERResult = result[0]
        print("FER Result: ", FERResult)
        FERval = FERResult

if __name__ == '__main__':

    webcam = cv2.VideoCapture(0)
    #print(webcam)
    mpFER = multiprocessing.Process(target=main, args=(webcam, model1, return_dict)) #Remove model2
    mpFER.start()

    with keyboard.Listener(
            on_release=spaceTrigger) as listener:
        listener.join()








