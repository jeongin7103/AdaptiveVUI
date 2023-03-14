# AdaptiveVUI
User-adaptive VUI: Multimodal signal based emotion recognition

This repository supports actual implementation of user-adaptive VUI which is equipped with keyword spotting system and emotion recognition engine. 
By utilizing facial and EEG data, we developed facial expression based emotion recognition (FER) model and EEG-based emotion recognition (EER) model.
FER model is developed using DAN (distract your attention) network, while EER model is developed using ShallowNet architecture. 

This code is available to acquire and stream the data (face, EEG) in real-time and run the keyword spotting function with the emotion inference engine in parallel. 

All given codes are executed on the Linux OS. 
# Quick start
  ### 1. Requirements
    pip install playsound
    pip install numpy
    pip install SpeechRecognition
    pip install multiprocess
    pip install pynput
    pip install ipython
    pip install matplotlib
   ### 2. Run the code
    sudo python3 vui_demo.py

# Result
The main code recognizes the user's speech and transcribes into a text to detect the keyword inside which is predetermined. 
Simultaneously, FER and EER inference code block predicts the user's emotion in real-time, resulting in probability score of 'satisfied' state. 
The predicted emotion is plotted in the emotion plane per each trial like this figure below. 


![image](https://user-images.githubusercontent.com/127823391/224917170-80c95283-6dea-4ed1-946a-4814c040d9d6.png)



# Reference
[1] Distract your attention: Multi-head cross attention network for facial expression recognition [[link]](https://arxiv.org/pdf/2109.07270.pdf)

[2] Deep learning with convolutional neural networks for EEG decoding and visualization [[link]](https://arxiv.org/abs/1703.05051)
