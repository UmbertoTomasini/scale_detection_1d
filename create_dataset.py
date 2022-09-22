import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import math
import matplotlib.pyplot as plt
import torch.optim as optim
import random
import itertools

def generate_dataset_T2( xi, gap_dist =1, img_size=32):
    assert gap_dist<xi 
    
    all_pos_1pt =  [] 
    
    #all possible positions 1 point
    for i in range(img_size):
        list_tmp = [i]
        all_pos_1pt.append(list_tmp)
    #combinations of all possible positions 2 points
    comb_all = list(itertools.combinations(all_pos_1pt,2))
    
    #labeling
    
    lab_all = []
    comb_all_gap = []
    for img in comb_all:
        pt1 = img[0]
        r1 = pt1[0]
        
        pt2 = img[1]
        r2 = pt2[0]
       
        if img[0] == img[1]: print(img)
        sec1 = (r1)//xi
        sec2 = (r2)//xi
        
        
        #different labelling      
        if sec1 == sec2:
            lab = +1
            lab_all.append(lab)
            comb_all_gap.append(img)
        else:
            lab= -1
            lab_all.append(lab)   
            comb_all_gap.append(img) 
    return comb_all_gap, lab_all

def generate_dataset_T1( xi, gap_dist =1, img_size=32):
    assert gap_dist<xi 
    all_pos_1pt =  [] 
    
    #all possible positions 1 point
    for i in range(img_size):
        list_tmp = [i]
        all_pos_1pt.append(list_tmp)
    #combinations of all possible positions 2 points
    comb_all = list(itertools.combinations(all_pos_1pt,2))
    
    #labeling
    
    lab_all = []
    comb_all_gap = []
    for img in comb_all:
        pt1 = img[0]
        r1 = pt1[0]
        
        pt2 = img[1]
        r2 = pt2[0]
       
        if img[0] == img[1]: print(img)
        d = min(abs(r2-r1),img_size-abs(r2-r1))
        
        if d<(xi-gap_dist):
            lab = +1
            lab_all.append(lab)
            comb_all_gap.append(img)
        elif d>=(xi+gap_dist):
            lab= -1
            lab_all.append(lab)   
            comb_all_gap.append(img)
            
    return comb_all_gap, lab_all
#----------------------------------------------------------------
#Main
#----------------------------------------------------------------
#NEEDING XI multiple of 2
xi = 2**6
gap_dist = 0
img_size = 2**8
#choose either task 1 or task 2
task = 2

if task==1:
    imgs, labs = generate_dataset_T1( xi=xi, gap_dist=gap_dist, img_size=img_size)
elif task==2:
    imgs, labs = generate_dataset_T21( xi=xi, gap_dist=gap_dist, img_size=img_size)
c = 0
imgs_plus = []
imgs_minus = []
for lab in labs:
    if lab>0:
        imgs_plus.append(imgs[c])
    else:
        imgs_minus.append(imgs[c])
    c+=1

imgs_plus = torch.tensor(imgs_plus)
imgs_minus = torch.tensor(imgs_minus)
print(imgs_plus.size())
print(imgs_minus.size())

torch.save(imgs_plus,"data_points_xi_%d_Circle_Gap_%.1f_PLUS_imgSize_%d_T_%d.pt" %(xi,gap_dist,img_size,task))
torch.save(imgs_minus,"data_points_xi_%d_Circle_Gap_%.1f_MINUS_imgSize_%d_T_%d.pt" %(xi,gap_dist,img_size,task))

