from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
import torch.nn.functional as F
import torch
from PIL import Image
from detector import detect_faces
from visualization_utils import show_results
import cv2
import numpy as np
import extract_feature_final as v1
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
import glob
from detection.RetinaFace.retinaface import RetinaFace
from detection.RetinaFaceAntiCov.retinaface_cov import RetinaFaceCoV
import threading
import concurrent.futures
import time
import torchvision.datasets as datasets

# trigger for thread-------------------------------------------
evt_snap = threading.Event()
evt_detect = threading.Event()
evt_extract = threading.Event()
evt_compare = threading.Event()

# dealing image------------------------------------------------
def de_preprocess(tensor):
    return tensor * 0.5 + 0.5

hflip = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

transform = transforms.Compose([
        transforms.Resize([int(128 * 112 / 112), int(128 * 112 / 112)]),  # smaller side resized
        transforms.CenterCrop([112,112]),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])

def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    hfliped_imgs = hflip(imgs_tensor)
    return hfliped_imgs

# Que for thread-----------------------------------------------
class Node:
    def __init__(self, data):
        self.data = data
        self.link = None

    def __str__(self):
        return str(self.data)

class Queue:
    def __init__(self, data):
        new_node = Node(data)
        self.front = new_node
        self.rear = new_node
        self.front.link = self.rear

    def __str__(self):
        print_queue = '<= [ '
        node = self.front
        while True:
            print_queue += str(node)
            if(node == self.rear):
                break
            try:
                node = node.link
            except:
                break
            print_queue += ', '
        print_queue += ' ] <='
        return print_queue

    def isEmpty(self):
        if self.front == self.rear:
            return True
        else:
            return False

    def enQueue(self, data):
        new_node = Node(data)
        self.rear.link = new_node
        self.rear = new_node

    def deQueue(self):
        if not self.isEmpty():
            node = self.front
            value = node.data
            self.front = self.front.link
            del node
            return value

    def peek(self):
        return self.front.data

# global value for image cropping------------------------------
crop_size = [112, 112]
thresh = 0.8
mask_thresh = 1.0
count = 1
gpuid = 0
image_path=""

# global value for model loading-------------------------------
detector = RetinaFace('/home/sylee/Face/face.evoLVe.PyTorch/detect/retinaface-R50/R50', 0, gpuid, 'net3')
#detector = RetinaFaceCoV('/home/sylee/Face/face.evoLVe.PyTorch/detect/mnet_cov2/mnet_cov2', 0, gpuid, 'net3l')
BACKBONE = IR_101([112,112])
BACKBONE_RESUME_ROOT = "/home/sylee/Face/CurricularFace/CurricularFace_Backbone.pth" 

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

# detecting function for first place----------------------------
def detect_stack(file_path):
    scales = [512, 980]
    size = [540, 980]

    try:
        frame = cv2.imread(file_path)

    except Exception as e:
        print("capture error", e)

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
    imgs = np.zeros([0,112,112,3])
    flips = np.zeros([0,112,112,3])
    if faces is not None:
        for i in range(faces.shape[0]):
            face = faces[i]
            box = face[0:4].astype(np.int)
            box_list.append(box)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            crop_face = frame[y1: y2, x1:x2]
            crop_face = cv2.resize(crop_face, (crop_size[0], crop_size[1]),
                               interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(crop_face, 'RGB')
            img = transform(img)
            fliped=hflip_batch(img)
            img=img.unsqueeze(0)
            fliped=fliped.unsqueeze(0) 
            if i == 0:
                imgs = img
                flips = fliped
            else:
                imgs = np.concatenate((imgs, img), axis=0)
                flips = np.concatenate((flips, fliped), axis=0)
    imgs = torch.from_numpy(imgs)
    flips = torch.from_numpy(flips)
    return imgs, flips

# detecting function for detecting thread at second place------------
def detect():
    while(True):
        evt_detect.wait()

        if(signal_que.peek()=='finish'):
            print("end detecting!!!")
            evt_extract.set()
            break

        x1_box_list = []
        y1_box_list = []
        scales = [512, 980]
        size = [540, 980]

        try:
            frame = image_que.peek()

        except Exception as e:
            print("capture error", e)

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

        frame2 = None
        box_list = []
        imgs = np.zeros([0,112,112,3])
        flips = np.zeros([0,112,112,3])
        if faces is not None:
            for i in range(faces.shape[0]):
                face = faces[i]
                box = face[0:4].astype(np.int)
                box_list.append(box)
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                x1_box_list.append(x1)
                y1_box_list.append(y1)
                frame2 = frame
                cv2.rectangle(frame2, (box[0], box[1]), (box[2], box[3]), (255,255,0), 2)
                crop_face = frame[y1: y2, x1:x2]
                crop_face = cv2.resize(crop_face, (crop_size[0], crop_size[1]),
                               interpolation=cv2.INTER_LINEAR)
                img = Image.fromarray(crop_face, 'RGB')
                img = transform(img)
                fliped=hflip_batch(img)
                img=img.unsqueeze(0)
                fliped=fliped.unsqueeze(0)
                if i == 0:
                    imgs = img
                    flips = fliped
                else:
                    imgs = np.concatenate((imgs, img), axis=0)
                    flips = np.concatenate((flips, fliped), axis=0)
            imgs = torch.from_numpy(imgs)
            flips = torch.from_numpy(flips)
   
            frame_const = frame2
            batch_que.enQueue(imgs)
            batch_que.deQueue()
            flip_que.enQueue(flips)
            flip_que.deQueue()
            frame_que.enQueue(frame_const)
            frame_que.deQueue()
            x_box_que.enQueue(x1_box_list)
            x_box_que.deQueue()
            y_box_que.enQueue(y1_box_list)
            y_box_que.deQueue()
        evt_detect.clear()
        evt_extract.set()

# feature comparing function----------------------------
def simple(A, B):
       temp =  np.subtract(A,B)
       ret = np.sqrt(np.sum(np.square(temp)))
       return ret

# getting feature for first place-----------------------
def get_feature_stack(image,fliped):
    return v1.extract_feature(image,fliped,BACKBONE,BACKBONE_RESUME_ROOT)

# getting feature for second place----------------------
def get_feature():
    while(True):
        evt_extract.wait()

        if(signal_que.peek()=='finish'):
            print("end extracting!!!")
            evt_compare.set()
            break

        image = batch_que.peek()
        fliped = flip_que.peek()
        gourp_const = np.zeros((image.shape[0],512))

        if image is None:
            print("wait for image")

        else:
            group_const = v1.extract_feature(image,fliped,BACKBONE,BACKBONE_RESUME_ROOT)
        feature_que.enQueue(group_const)
        feature_que.deQueue()
        evt_extract.clear()
        evt_compare.set()

# for group matching----------------------------------
mean_const = [0.0,0.0,0.0,0.0]
evaluate_const =[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
count_const = 0
def group():
    global count_const
    global evaluate_const
    global mean_const
    while(True):
        evt_compare.wait()

        if(signal_que.peek()=='finish'):
            print("end comparing!!!")
            break

        frame_const=frame_que.peek()
        x1_box_list=x_box_que.peek()
        y1_box_list=y_box_que.peek()
        group_const=feature_que.peek()
        for i, im1 in enumerate(group_const.cpu().detach().numpy()):
            low = 2.0
            num = 0
            for k, im2 in enumerate(standard.cpu().detach().numpy()):
                temp = simple(im1,im2)
                evaluate_const[i][k] += temp
                if temp<low:
                    low = temp
                    num = k
            if low<0.92:
                frame_const = cv2.putText(frame_const, "person{}".format(num+1), (x1_box_list[i], y1_box_list[i]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            mean_const[i] += low
        count_const += 1
        #cv2.imshow('person1',person_img[0])
        #cv2.imshow('person2',person_img[1])
        #cv2.imshow('person3',person_img[2])
        #cv2.imshow('person4',person_img[3])
        
        try:
            frame_const = cv2.resize(frame_const, (512, 1024), interpolation = cv2.INTER_CUBIC)
            cv2.imshow('VideoCapture', frame_const)
        except Exception as e:
            print(str(e))
        if cv2.waitKey(1) > 0 :
            break
        
        evt_compare.clear()
        evt_snap.set()

# for video------------------------------------------
def snap_video():
    path = path_que.peek()
    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(* 'XVID')
    image_que.deQueue()

    while(True):
        evt_snap.clear()
        ret, video = cap.read()
        if ret == False or signal_que.peek()=='finish':
            cap.release()
            signal_que.enQueue('finish')
            signal_que.deQueue()
            evt_detect.set()
            break

        if video.all() != None and int(cap.get(1)) % 5 == 0:
            image_que.enQueue(video)
            image_que.deQueue()
            evt_detect.set()
            evt_snap.wait()

        else:
            image_que.enQueue(np.zeros((112,112,3)))
            image_que.deQueue()

# main-----------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "compare images")
    parser.add_argument("-point1_image_root", "--point1_image_root", help = "specify your source dir", default = "/home/sylee/Face/test/detected/trump_mask/trump/trump", type = str)
    parser.add_argument("-point2_image_root", "--point2_image_root", help = "specify your source dir", default = "/home/sylee/Face/test/detected/trump_mask/trump/trump", type = str)
    parser.add_argument("-num_point1", "--num_point1", help = "specify your image number in file", default = 0, type = int)
    parser.add_argument("-num_point2", "--num_point2", help = "specify your image number in file", default = 1, type = int)
    args = parser.parse_args()
    point1_image_root = args.point1_image_root
    point2_image_root = args.point2_image_root
    num_point1 = args.num_point1
    num_point2 = args.num_point2
    person_img,person_fliped = detect_stack("/home/sylee/Face/test_mask_group.jpg")
    standard = get_feature_stack(person_img,person_fliped)
    
    image_que = Queue(np.zeros((112,112,3)))
    signal_que = Queue("alive")
    x_box_que = Queue([])
    y_box_que = Queue([])
    frame_que = Queue(np.zeros((112,112,3)))
    batch_que = Queue(np.zeros((1,112,112,3)))
    flip_que = Queue(np.zeros((1,112,112,3)))
    feature_que = Queue(np.zeros((0,0)))    
    
    path_que = Queue("/home/sylee/Face/mask.mp4")
    video_thread = threading.Thread(target=snap_video,name='video')
    detecting_thread = threading.Thread(target=detect,name='detection')
    extracting_thread = threading.Thread(target=get_feature,name='extraction')
    comparing_thread = threading.Thread(target=group,name='comparing')
    
    init_time = time.time() 
    video_thread.start()
    detecting_thread.start()
    extracting_thread.start()
    comparing_thread.start()
    
    video_thread.join()
    detecting_thread.join()
    extracting_thread.join()
    comparing_thread.join()    
    
    print("finish: ",time.time()-init_time)
    print("mean-point1:{},point2:{},point3:{},point4:{}".format(mean_const[0]/count_const,mean_const[1]/count_const,mean_const[2]/count_const,mean_const[3]/count_const))
    for i,value in enumerate(evaluate_const):
        print("analye {}: {}, {}, {}, {}".format(i+1,value[0]/count_const,value[1]/count_const,value[2]/count_const,value[3]/count_const))
    cv2.destroyAllWindows()
