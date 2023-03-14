import torch
import torch.nn as nn
import torch.utils.data as data

from deepface import DeepFace
from torchvision import transforms
from numpy import random
from PIL import Image
import argparse
from collections import OrderedDict
import numpy as np
#import networks.resnet as ResNet
from danMtlTrt.networks.Res50 import res50
from torch2trt import torch2trt
import time 
import cv2

#from pynput.keyboard import Key, Controller
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='./00001.jpg', help='image path.')
    parser.add_argument('--model1', type=str, default='./models/SSM1_pair_6_18_epoch_0.993074472505656_f1_3_fold_ResNet50.pth', help='요인1 model path.')
    parser.add_argument('--model2', type=str, default='./models/SSM1_pair_7_12_epoch_1.0_f1_3_fold_ResNet50.pth', help='요인2 model path.')
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')

    return parser.parse_args()
include_top = True
N_IDENTITY = 8631
class inference():
    def __init__(self,num_head,model_path):
        if torch.cuda.is_available():
            print("GPU activate")        
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.model = res50()
        self.data_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        #checkpoint = torch.load(model_path)
        #if isinstance(self.model, nn.DataParallel): # GPU 병렬사용 적용 
        #    self.model.load_state_dict(checkpoint['model_state_dict'], strict=True) 
        #else: # GPU 병렬사용을 안할 경우 
        #    state_dict = checkpoint['model_state_dict']
        #    new_state_dict = OrderedDict() 
        #    for k, v in state_dict.items(): 
        #        name = k[7:] # remove `module.` ## module 키 제거 
        #        new_state_dict[name] = v 
        #    self.model.load_state_dict(new_state_dict, strict = True)
        
       
        self.model.to(self.device)
        self.model.eval()
        x = torch.rand(1,3,224,224)
        self.model(x)

    def pred(self,img):
        #print("start")

        try:
            print(np.shape(img))
        except Exception as e:
            print(e)
        #print("start")
        x = self.data_transforms(img)
        x = x.to(self.device)
        x = x.view(1,3,224,224)

        #print("start")
        # print("original start")
       # start_o = time()
       # out_expr= self.model(x)
       # end_o =time()
       # print("original time : %.5f"%(end_o-start_o))
       
       
        model_trt = self.model

        #time.process_time()
        #start_t = time.process_time()
        out= model_trt(x)
        #end_t =time.process_time()

        return out.detach().cpu().numpy()
def calctime(start,end):
    start = start.split(":")

    s_H, s_M, s_S = int(start[0]), int(start[1]), int(start[2])
    end = end.split(":")

    e_H, e_M, e_S = int(end[0]),int(end[1]),int(end[2])  
    H, M, S = e_H-s_H, e_M-s_M, e_S-s_S
    T= (3600*e_H+60*e_M+e_S) - (3600*s_H+60*s_M+s_S)
    timelist= []
    for i in range(0,T):
        n_S = s_S+i
        
        n_M = s_M+ n_S//60
        n_S = n_S%60
        n_H = s_H+ n_S//60
        n_M = n_M%60
        timelist.append("%s:%s:%s"%(str(n_H).zfill(2),str(n_M).zfill(2),str(n_S).zfill(2)) ) 
    #print(timelist)
    return timelist
def main(webcam,model1,dic):

    end = 0.0
    while webcam.isOpened():

        status, frame = webcam.read()

        if status:
            cv2.imshow("test", frame)
            if time.time() - end >= 1:
                cv2.imshow("pred", frame)
                img = Image.fromarray(frame.copy())
                img.save("img.jpg")

                end = time.time()

        img=Image.fromarray(frame.copy())

        img.save("img.jpg")
        #print("save")
        face = DeepFace.detectFace(img_path = "img.jpg",
                target_size = (224, 224),
                detector_backend = 'opencv',
                enforce_detection = False

                )
        if cv2.waitKey(1) == 27:
            break

        face = Image.fromarray(face.astype(np.uint8))

        dic[time.strftime('%H:%M:%S', time.localtime(time.time()))] = face
        #time.sleep(1)


if __name__ == "__main__":
    dic ={}
    main(dic)