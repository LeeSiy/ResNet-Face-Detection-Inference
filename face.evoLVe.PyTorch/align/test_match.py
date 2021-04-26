from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
import torch.nn.functional as F
import torch
from PIL import Image
from detector import detect_faces
from visualization_utils import show_results
import cv2
import numpy as np
import extract_feature_v1 as v1
from backbone.model_resnet import ResNet_101
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from numpy import dot
from numpy.linalg import norm
import numpy as np
import os
import argparse

def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))

def simple2(A, B):
       return abs(np.mean(A-B))

def simple(A, B):
       temp =  np.subtract(A,B)
       ret = (1-np.sum(np.square(temp))/4)*100
       return ret

def compare(model_path, image_path1, image_path2):
    BACKBONE = IR_101([112,112])
    BACKBONE_RESUME_ROOT = model_path
    if BACKBONE_RESUME_ROOT:
        print("=" * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT),strict=False)
        else:
            print("No Checkpoint Found at '{}'".format(BACKBONE_RESUME_ROOT))
            exit()
        print("=" * 60)
    import time
    BACKBONE.cuda()
    image1 = v1.extract_feature(image_path1,BACKBONE,model_path)
    image2 = v1.extract_feature(image_path2,BACKBONE,model_path)
    return simple(image1[0],image2[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "compare images")
    parser.add_argument("-image1_root", "--image1_root", help = "specify your source dir", default = "/home/sylee/Face/test/detected/test_mask/test", type = str)
    parser.add_argument("-image2_root", "--image2_root", help = "specify your source dir", default = "/home/sylee/Face/test/detected/test_mask/test", type = str)
    parser.add_argument("-model_root", "--model_root", help = "specify your model path", default = "/home/sylee/Face/CurricularFace/CurricularFace_Backbone.pth", type = str)
    args = parser.parse_args()

    image1_root = args.image1_root
    image2_root = args.image2_root
    model_root = args.model_root 
    
    res = compare(model_root, image1_root, image2_root)
    print("image similarity {}%".format(res))

