from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
#from backbone.model_mobilefacenet import MobileFaceNet
import torch.nn.functional as F
import torch
from PIL import Image
from detector import detect_faces
from visualization_utils import show_results
import cv2
import numpy as np
import extract_feature_v2 as v2
import extract_feature_v1 as v1
from backbone.model_resnet import ResNet_101
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from numpy import dot
from numpy.linalg import norm
import numpy as np
import os

def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))

def simple2(A, B):
       return abs(np.mean(A-B))

def simple(A, B):
       temp =  np.subtract(A,B)
       ret = (1-np.sum(np.square(temp))/4)*100
       return ret  
class TempModel(nn.Module):
    def __init__(self):
        super(TempModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, (3, 3))
    def forward(self, inp):
        return self.conv1(inp)

model_path = '~/Face/CurricularFace/CurricularFace_Backbone.pth'

img_path_trump = '~/Face/test/detected/trump_mask/trump'
img_path_biden = '~/Face/test/detected/biden_mask/biden'
img_path_elon = '~/Face/test/detected/elon_mask/elon'
img_path_hugh = '~/Face/test/detected/hugh_mask/hugh'
img_path_test = '~/Face/test/detected/test_mask/test'

#-------------------------------------------------------------------------
'''
img_biden = Image.open('~/Face/test/mask/test_mask/test/biden.jpg')
bounding_boxes, landmarks = detect_faces(img_biden) 
img_biden = show_results(img_biden, bounding_boxes, landmarks)  
img_biden = np.array(img_biden)
cv2.imwrite('./result/result_mask_biden.png',img_biden)

model_path = '~/Face/CurricularFace/CurricularFace_Backbone.pth'
img_trump = Image.open('~/Face/test/mask/test_mask/test/trump.jpg')
bounding_boxes2, landmarks2 = detect_faces(img_trump)
img_trump = show_results(img_trump, bounding_boxes2, landmarks2)
img_trump = np.array(img_trump)
cv2.imwrite('./result/result_mask_trump.png',img_trump)
'''
#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
BACKBONE = IR_101([112,112])
BACKBONE_RESUME_ROOT = '~/Face/CurricularFace/CurricularFace_Backbone.pth'
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
for i in range(1):
    a= time.time()
    trump = v1.extract_feature(img_path_trump,BACKBONE,model_path)
    print("time pass",time.time()-a)
biden = v1.extract_feature(img_path_biden,BACKBONE,model_path)
elon = v1.extract_feature(img_path_elon,BACKBONE,model_path)
hugh = v1.extract_feature(img_path_hugh,BACKBONE,model_path)
test = v1.extract_feature(img_path_test,BACKBONE,model_path)
#-------------------------------------------------------------------------
count = 0
answer = 'who'
min_n = 0.0

print("---------------------------------start test-----------------------------------")
print("test list index\n0.Biden\n1.Biden\n2.Elon\n3.Hugh\n4.Hugh\n5.Trump\n6.Trump\n7.Unknown")


#Trump
id_n = 0
print("\n*****Case1: test images compared with Trump*****")
for i, te in enumerate(test):
    for tr in trump:
        print("Group Trump",i, simple(te,tr))
        if simple(te,tr)>min_n:
            min_n = simple(te,tr)
            id_n = i
    print()
print("{}th list is highest, point = {}".format(id_n,min_n))
#Biden
min_n = 0.0
id_n = 0
print("\n*****Case2: test images compared with Biden*****")
for i, te2 in enumerate(test):
    for bi in biden:
        print("Group Biden",i, simple(bi,te2))
        if simple(te2,bi)>min_n:
            min_n = simple(bi,te2)
            id_n = i
    print()   
print("{}th list is highest, point = {}".format(id_n,min_n))


#Elon
min_n = 0.0
id_n = 0
print("\n*****Case3: test images compared with Elon*****")
for i, te3 in enumerate(test):
    for el in elon:
        print("Group Elon",i, simple(el,te3))
        if simple(te3,el)>min_n:
            min_n = simple(el,te3)
            id_n = i
    print()
print("{}th list is highest, point = {}".format(id_n,min_n))


#Hugh
min_n = 0.0
id_n = 0
print("\n*****Case4: test images compared with Hugh*****")
for i, te4 in enumerate(test):
    for hu in hugh:
        print("Group Hugh",i, simple(hu,te4))
        if simple(te4,hu)>min_n:
            min_n = simple(hu,te4)
            id_n = i
    print()
print("{}th list is highest, point = {}".format(id_n,min_n))

