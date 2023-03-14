import time
import keyboard
import matplotlib.pyplot as plt
import speech_recognition as sr
from playsound import playsound
import random
import multiprocessing
import cv2
import numpy as np
from danMtl.inference50_t import main, calctime,inference
from IPython.display import display, clear_output
from pynput.keyboard import Key, Listener
from RDALatest11302022 import mainRDAEEG
from BRL_EEG_python_timestamp import eeg_predict
from datetime import datetime

from pynput import keyboard

global emptylist
TimeStamps = []
TimeStampsEEG = [datetime.strftime(datetime.now(),'%H:%M:%S.%f'),datetime.strftime(datetime.now(),'%H:%M:%S.%f')]
triggerGlobal = False

manager = multiprocessing.Manager()
return_dict = manager.dict()
model1 = inference(4, "/VUI experiment/danMtlTrt/models/resnet50_ft_weight.pkl")  # model2 path
print("model 1 load")
print("done")


FERval = [float(),float()]
EEGval = float()




def emotionPlotting():
    global curQuadrant
    global fig
    global EEGval # Provided by EEG inference
    global FERval # Provided by FER Inference

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # while True:
    plt.xlabel("Trustability")
    plt.ylabel("Stability")
    plt.xticks([0, 0.5, 1])
    plt.yticks([0, 0.5, 1])
    plt.axvline(x=0.5)
    plt.axhline(y=0.5)
    ax.set_xlim([0.0, 1])
    ax.set_ylim([0.0, 1])

    a = random.uniform(0.0, 1.0)
    b = 1 - a
    FERVal = [a, b]

    EEGVal = random.uniform(0.0, 1.0)
    if FERVal[0] > FERVal[1]:
        x = 0
    else:
        x = 1
    y = round(EEGVal)

    print("FER Probability: ", x, " EEG Probability: ", y)

    if x == 1 and y == 1:
        print("Q1")
        curQuadrant[0] = FERVal[1] + EEGVal
        curQuadrant[1] = "Q1"
        rectangle = plt.Rectangle((0.5, 0.5), 0.5, 0.5, fc='blue', ec="red")
        ax.add_patch(rectangle)
    elif x == 0 and y == 1:
        print("Q2")
        curQuadrant[0] = FERVal[1] + EEGVal
        curQuadrant[1] = "Q2"
        rectangle = plt.Rectangle((0.0, 0.5), 0.5, 0.5, fc='blue', ec="red")
        ax.add_patch(rectangle)
    elif x == 0 and y == 0:
        print("Q3")
        curQuadrant[0] = FERVal[1] + EEGVal
        curQuadrant[1] = "Q3"
        rectangle = plt.Rectangle((0.0, 0.0), 0.5, 0.5, fc='blue', ec="red")
        ax.add_patch(rectangle)
    elif x == 1 and y == 0:
        print("Q4")
        curQuadrant[0] = FERVal[1] + EEGVal
        curQuadrant[1] = "Q4"
        rectangle = plt.Rectangle((0.5, 0.0), 0.5, 0.5, fc='blue', ec="red")
        ax.add_patch(rectangle)

    ax.plot()
    display(fig)
    plt.pause(1)
    clear_output(wait=True)

def spaceTrigger(key):#key):



    if key == keyboard.Key.space:
        # Stop listener
        print("Starting Inference")
        KWS()

    if key == keyboard.Key.esc:
        # Stop listener
        return False
    pass



def inputVoice(lang):
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        playsound('/VUI experiment/beep.mp3')
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
    with open('/VUI experiment/kwlist.txt') as f:
        kl = f.read().splitlines()
        #print(kl)
    return kl

def timeStampRec(startTime, EndTime):
    timeLog = open("/VUI experiment/timestamp.txt", "w")
    timeLog.write(startTime, EndTime)
    pass

def KWS():
    plt.close() # Reinitialize plot
    global TimeStamps
    global triggerGlobal
    global TimeStampsEEG

    userQuestion = inputVoice("ko")
    keyword_list = txtkeywordList()
    superKeyword = set(keyword_list) & set(userQuestion)

    if superKeyword == set():

        playsound('/VUI experiment/null.mp3')
        return
    else:
        supkey = list(superKeyword)
        if supkey[0] in userQuestion:
            vuiResponse = responseGenerator()

            for kw in txtkeywordList():
                if supkey[0] == kw:
                    path = '/VUI experiment/respo/' + vuiResponse + "/" + supkey[0] + '.mp3'
                    startTime = time.strftime('%H:%M:%S', time.localtime(time.time()))
                    startTimeEEG = datetime.strftime(datetime.now(),'%H:%M:%S.%f')

                    playsound(path, block=True)
                    print("End reply")
                    endTime = time.strftime('%H:%M:%S', time.localtime(time.time()))
                    endTimeEEG = datetime.strftime(datetime.now(),'%H:%M:%S.%f')
                    TimeStampsEEG = [startTimeEEG,endTimeEEG]
                    print("KWS EEG Time: ", TimeStampsEEG)
                    # FER timestamp
                    TimeStamps = [startTime, endTime]
                    EEGInfer()
                    time.sleep(1)


def EEGInfer():
    global TimeStampsEEG
    print("RDA Start Time: ",RDAstarttime)
    eegresult = eeg_predict(RDAstarttime,TimeStampsEEG[0],TimeStampsEEG[1])

    print("EEG Result: ",eegresult)
    return


def FERinfer():
    global model1
    global return_dict
    global manager
    global TimeStamps
    #global model2
    print("FER infer")

    #try:

    print("Use This: ", TimeStamps)
    li = calctime(TimeStamps[0], TimeStamps[1])
    start = time.time()
    print(return_dict)
    ndic = return_dict.copy()

    result = np.array([[0.0, 0.0]])

    i = li[-1]

    if i in ndic:
        print("i: ", i)
        print(ndic)

        result[0] = model1.pred(ndic[i])  # axis 1 #time is key
        print("RESULT: ",result)
        FERResult = result[0]


        print("Axis 1: ", FERResult)
        print("model 1: ", FERResult)



if __name__ == '__main__':

    mpRDA = multiprocessing.Process(target=mainRDAEEG)
    mpRDA.start()
    RDAstarttime = datetime.strftime(datetime.now(),'%H:%M:%S.%f')
    KWS()
    FERinfer()
    #mpFER = multiprocessing.Process(target=main, args=(webcam, model1, return_dict))



    with keyboard.Listener(
            on_release=spaceTrigger) as listener:
        listener.join()








