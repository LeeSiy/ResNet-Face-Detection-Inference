# Helper function for extracting features from pre-trained models
import torch
from PIL import Image
import numpy as np
import os
import time

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

def extract_feature(img_array,flip_array, backbone, model_root, embedding_size = 512, batch_size = 512, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta = True):

    assert (os.path.exists(model_root))
    NUM_CLASS = len(img_array) 
    backbone.to(device)
    backbone.eval() 
    features = np.zeros([NUM_CLASS,512])
    #time_init = time.time()
    emb_batch = backbone(img_array.to(device)).cpu() + backbone(flip_array.to(device)).cpu()
    with torch.no_grad():
        features = l2_norm(emb_batch)
        #print("delay for extraction: ".format(time.time()-time_init))
    return features
