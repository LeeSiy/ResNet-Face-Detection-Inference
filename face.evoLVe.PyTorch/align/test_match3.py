from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
import torch.nn.functional as F
import torch
from PIL import Image
from detector import detect_faces
from visualization_utils import show_results
import cv2
import numpy as np
import extract_feature_v4 as v1
from backbone.model_resnet import ResNet_101
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from numpy import dot
from numpy.linalg import norm
import numpy as np
import os
import argparse
import torchvision.transforms as transforms
import sys
import datetime
import glob
from detection.RetinaFace.retinaface import RetinaFace
import time

#-------------------------------------------------------------------------------------------------------
crop_size = [112, 112]
thresh = 0.8
mask_thresh = 1.0
count = 1
gpuid = 0
model1_init = time.time()
detector = RetinaFace('your/path/retinaface-R50/R50', 0, gpuid, 'net3')
print("Time for loading RetinaFace: {}", time.time() - model1_init)
BACKBONE = IR_101([112,112])
model2_init = time.time()
BACKBONE_RESUME_ROOT = 'yout/path/CurricularFace_Backbone.pth' 
if BACKBONE_RESUME_ROOT:
    print("=" * 60)
    if os.path.isfile(BACKBONE_RESUME_ROOT):
        print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
        BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT),strict=False)
    else:
        print("No Checkpoint Found at '{}'".format(BACKBONE_RESUME_ROOT))
        exit()
    print("=" * 60)
BACKBONE.cuda()
print("Time for loading CurricularFace: {}", time.time() - model2_init)

#-------------------------------------------------------------------------------------------------------
    
def detect(file_path):
    a = time.time()
    scales = [512, 980]
    size = [540, 980]
    try:
        frame = cv2.imread(file_path)

    except Exception as e:
        print("capture error", e)
        #continue

    frame_shape = frame.shape
    target_size = scales[0]
    max_size = scales[1]

    im_size_min = np.min(frame_shape[0:2])
    im_size_max = np.max(frame_shape[0:2])

    im_scale = float(target_size) / float(im_size_min)

    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    scales = [im_scale]
    flip = False

    for c in range(count):
        faces, landmarks = detector.detect(frame,
                                       thresh,
                                       scales=scales,
                                       do_flip=flip)

    box_list = []
    crop_face_batch = None
    if faces is not None:
        for i in range(faces.shape[0]):

            face = faces[i]
            box = face[0:4].astype(np.int)
            box_list.append(box)

            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255,255,0), 2)
            crop_face = frame[y1: y2, x1:x2]
            crop_face = cv2.resize(crop_face, (crop_size[0], crop_size[1]),
                               interpolation=cv2.INTER_LINEAR)
            
            if i == 0:
                crop_face_batch = np.expand_dims(crop_face, axis=0)
            else:
                crop_face = np.expand_dims(crop_face, axis=0)
                crop_face_batch = np.concatenate((crop_face_batch, crop_face), axis=0)

    return crop_face_batch, frame
#------------------------------------------------------------------------------

def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))

def simple2(A, B):
       return abs(np.mean(A-B))

def simple(A, B):
       temp =  np.subtract(A,B)
       ret = (1-np.sum(np.square(temp))/4)*100
       return ret

def get_feature(model_path, image):
    image = v1.extract_feature(image,BACKBONE,model_path)
    return image

def get_images(file_path):
    file_list = os.listdir(file_path)
    image_list = [file for file in file_list if file.endswith(".jpg")]

    images = np.zeros([1,112,112,3])
    for i,img in enumerate(image_list):
        image = detect(os.path.join(file_path,img))
        if i == 0:
            images = image
        else:
            images = np.append(images, image.reshape(1,112,112,3),axis=0)
    return images, image_list

def single_match(img1, single):
    high = 0.0
    num = 0
    for i, im in enumerate(img1):
        temp = simple(im,single)
        if temp>high:
            high = temp
            num = i     
    return num
def one_on_one_match(img1,img2):
    res = simple(img1,img2)
    print("image similarity {}%".format(res))

def group(img1,img2):
    list_match = []
    for i, im1 in enumerate(img1):
        high = 0.0
        num = 0
        for k, im2 in enumerate(img1):
            temp = simple(im1,im2)
            if temp>high:
                high = temp
                num = k
        list_match.append(num) 
    return list_match
#------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "compare images")
    parser.add_argument("-point1_image_root", "--point1_image_root", help = "specify your source dir", default = "", type = str)
    parser.add_argument("-point2_image_root", "--point2_image_root", help = "specify your source dir", default = "", type = str)
    parser.add_argument("-num_point1", "--num_point1", help = "specify your image number in file", default = 0, type = int)
    parser.add_argument("-num_point2", "--num_point2", help = "specify your image number in file", default = 1, type = int)
    args = parser.parse_args()

    point1_image_root = args.point1_image_root
    point2_image_root = args.point2_image_root
    num_point1 = args.num_point1
    num_point2 = args.num_point2

    group, group_box = detect("")
    person, person_box = detect("")
    group_res = get_feature(BACKBONE_RESUME_ROOT, group)
    person_res = get_feature(BACKBONE_RESUME_ROOT, person)
    num = single_match(group_res,person_res)
    
    cv2.imshow('group',group_box)
    cv2.imshow('result',group[num])
    cv2.imshow('person',person[0])
    cv2.waitKey(8000)
    cv2.destroyAllWindows()

    print("{}th person".format(num+1))        

    '''
    point1_images, point1_list = get_images(point1_image_root)
    point1_res = get_feature(BACKBONE_RESUME_ROOT, point1_images)
    print("point1 complete!!!")
    
    point2_images, point2_list = get_images(point2_image_root)
    point2_res = get_feature(BACKBONE_RESUME_ROOT, point2_images)
    print("point2 complete!!!")
    
    print("\n========================one on one testing==========================")
    one_on_one_match(point1_res[num_point1],point2_res[num_point2])
        
    print("\n==========================group testing==========================")
    list_group_match = group(point1_res,point2_res)
    for i,name in enumerate(point1_list):
        temp = list_group_match[i]
        print("{} is assumed to be {}.".format(name,point2_list[temp]))
    '''
