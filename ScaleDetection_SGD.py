import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import math
import torch.optim as optim
from random import randrange
import random
from convolutional_nets import*
import sys
import numpy as np
import copy

def HingeLoss(A, y,o):
        m = nn.ReLU()
        return (A**(-1))*m(1-y*o)

def w_norm(net,p,H,nbl):

        nrm = 0.
        len_par = 0
        for mm in range(nbl):
            par = net.conv[mm].w                                   
            nrm += par.abs().pow(p).sum()  #.pow(1/p)
        len_par = (H**(p/2))*nbl                              
        
        return nrm /len_par
        
def Loss01(y,o):
        ntot = len(o)
        oy = o*y
        nwrong = len(oy[oy<0])
        
        return nwrong/ntot

def from_coord_to_img (pts, img_size, magn):
        img = torch.zeros((img_size))
        img[int(pts[0,0])]= (magn)**.5
        img[int(pts[1,0])]= (magn)**.5
        return img


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, flush = True)
xi = int(sys.argv[1])
img_size = int(sys.argv[2])
gap = float(sys.argv[3])
nbl = int(sys.argv[4])
LR = float(sys.argv[5])
H = int(sys.argv[6])
magn = float(sys.argv[7])
ridge = float(sys.argv[8])
pv = int(sys.argv[9])

#choose either task 1 or 2
task=2

data_file_plus= torch.load("data_points_xi_%d_Circle_Gap_%.1f_PLUS_imgSize_%d_T_%d.pt" %(xi,gap_dist,img_size,task)).to(device)
data_file_minus = torch.load("data_points_xi_%d_Circle_Gap_%.1f_MINUS_imgSize_%d_T_%d.pt" %(xi,gap_dist,img_size,task)).to(device)
nplus = data_file_plus.size()[0] 
nminus = data_file_minus.size()[0] 
indp = torch.randperm(nplus)
indm = torch.randperm(nminus)
data_file_plus = data_file_plus[indp,:,:]
data_file_minus = data_file_minus[indm,:,:]


NTOT = 2*nplus
data_file =torch.zeros((NTOT,1,2,1)).to(device)
data_file[:int(NTOT/2),0,:,:] = data_file_plus[:int(NTOT/2),:,:]
data_file[int(NTOT/2):,0,:,:] = data_file_minus[:int(NTOT/2),:,:]

lab_file = torch.ones((NTOT)).to(device)
lab_file[:int(NTOT/2)] = -lab_file[:int(NTOT/2)]

#Choosing:
#different ntrain training points,
#nval validation points, 
#ntest test points
pp = np.array([1024])
ntrain = int(np.sum(pp))
nval = 128
ntest = NTOT - ntrain -nval


fsize = 2
stride = 2

idx_strides = [i for i in range(nbl)]

#------------------------------------
data_train =torch.zeros((ntrain,1,2,1)).to(device)
data_train[:int(ntrain/2),0,:,:] = data_file[:int(ntrain/2),0,:,:] 
data_train[int(ntrain/2):,0,:,:] = data_file[int(NTOT/2):int(NTOT/2)+int(ntrain/2),0,:,:] 

lab_train = torch.ones((ntrain)).to(device)
lab_train[:int(ntrain/2)] = -lab_train[:int(ntrain/2)]

indt = torch.randperm(ntrain)
data_train = data_train[indt,:,:,:]
lab_train =  lab_train[indt]
#------------------------------------

data_val =torch.zeros((nval,1,2,1)).to(device)
data_val[:int(nval/2),0,:,:] = data_file[int(ntrain/2):int(ntrain/2)+int(nval/2),0,:,:] 
data_val[int(nval/2):,0,:,:] = data_file[int(NTOT/2)+int(ntrain/2):int(NTOT/2)+int(ntrain/2)+int(nval/2),0,:,:] 

labval = torch.ones((nval)).to(device)
labval[:int(nval/2)] = -labval[:int(nval/2)]

indv = torch.randperm(nval)
data_val = data_val[indv,:,:,:]
labval =  labval[indv]

#------------------------------------
data_test =torch.zeros((ntest,1,2,1)).to(device)
data_test[:int(ntest/2),0,:,:] = data_file[int(ntrain/2)+int(nval/2):int(NTOT/2),0,:,:]
data_test[int(ntest/2):,0,:,:] = data_file[int(NTOT/2)+int(ntrain/2)+int(nval/2):,0,:,:]

labtest = torch.ones((ntest)).to(device)
labtest[:int(ntest/2)] = -labtest[:int(ntest/2)]

indt = torch.randperm(ntest)
data_test = data_test[indt,:,:,:]
labtest =  labtest[indt]
#------------------------------------------

image_test = torch.zeros([ntest,1,img_size]).to(device)
lab_test = torch.zeros([ntest]).to(device)

for j in range(ntest):
    
    img = from_coord_to_img (data_test[j,0,:,:], img_size,magn)
    image_test[j,0,:] = img
    lab_test[j] = labtest[j]


image_val = torch.zeros([nval,1,img_size]).to(device)
lab_val = torch.zeros([nval]).to(device)

for j in range(nval):

    img = from_coord_to_img (data_val[j,0,:,:], img_size, magn)
    image_val[j,0,:] = img
    lab_val[j] = labval[j]


sum_P = 0
for P in pp:

    batch_sgd = 8
    
    print("#############################################", flush = True)
    X_tr = torch.zeros([P,1,img_size]).to(device)
    y_tr = torch.zeros([P]).to(device)
    for j in range(P):
        img = from_coord_to_img (data_train[j+sum_P,0,:,:], img_size, magn)
        X_tr[j,0,:] = img
        y_tr[j] = lab_train[j+sum_P]
     
    sum_P+= P
  
    num_nets = 10 
    lcurve = torch.ones(num_nets).to(device) #at end of training
    qq = 1
    
    while (qq-1) < num_nets:

        train_epochs = []
        test_epochs = []
        w_epochs = []
        w_epochs_layers = []
        w_reg = []
        torch.manual_seed(qq)
        net = ConvNetGAPMF_1d(alpha_tilde = A, n_blocks= nbl , input_ch=1, h=H,  idx_strides = idx_strides, filter_size=fsize, stride_int=stride, out_dim=1).to(device)
        net0 = copy.deepcopy(net)
        torch.save(net0.state_dict(),r"models_save_1d/gap_P_%d_xi_%d_nbl_%d_aTilde_%d_LR_%.6f_epoch_0_gap_%.1f_iter_%d_H_%d_magn_%.1f_rdg_%.6f_pv_%d_str_%d.pt"%(P,xi,nbl,A_exp,LR,gap,qq,H,magn,ridge,pv,stride))
        
        pars = net0.parameters()
        w0 = [] 
        for par in pars:
            w0.append(par)
    
        den = 0
        for g_idx in range(len(w0)):
            den += ((w0[g_idx])**2).mean()
         
        mom = 0.
        optimizer = optim.SGD(net.parameters(), lr= LR*H, momentum=mom)
        max_epochs = 6* (10**5)
        epoch_interpol = max_epochs 
        flag = 0
        for epoch in range(max_epochs):  # loop over the dataset multiple times
        
            ind = torch.randperm(P)
            X_tr = X_tr[ind]
            y_tr = y_tr[ind]
            trLossWeightEpoch = 0 
            trLossEpoch = 0
            nt = int(P/batch_sgd)
            for kk in range(int(P/batch_sgd)):
                if(epoch%10==0 and kk%99==0): 
                    print("--------------------------------", flush = True)
                    print("Number epoch: "+str(epoch)+", Number batch: "+str(kk), flush = True)
                batch = X_tr[kk*batch_sgd : (kk+1)*batch_sgd]
                ybatch = y_tr[kk*batch_sgd : (kk+1)*batch_sgd]
                optimizer.zero_grad()

                # forward + backward + optimize
                
                outputs = net(batch).to(torch.float32)
                if outputs.mean().isnan():
                    print("NAN!!", flush = True)
                    break                

                loss = HingeLoss(A, outputs.squeeze(), ybatch.squeeze())#criterion(outputs, labels)
                
                loss = loss.mean()
                loss_interpol = loss.item()
                if pv>0: loss += ridge*w_norm(net,p=pv,H=H,nbl = nbl)
                trLossEpoch += loss_interpol            
                trLossWeightEpoch += loss
                loss.backward()
                optimizer.step()

            
            with torch.no_grad():
                if epoch%10 ==0:
                    #saving test error
                    with torch.no_grad():
                        L=0
                        
                        batch_test = 8
                        NB = int(nval/batch_test)
                        for hh in range(NB):
                            it = image_val[hh*batch_test : (hh+1)*batch_test,:,:]
                            lt = lab_val[hh*batch_test : (hh+1)*batch_test]
                            out = net(it)
                            L+=Loss01(out.squeeze(), lt.squeeze())
                        L = L/NB
                        vali = L 
                    test_epochs.append(L)
                    print("Val error for P: "+str(P)+" and epoch: "+str(epoch)+" is: "+str(L), flush = True)
                
                    #evolution weights
                    pars = net.parameters()
                    w = []
                    for par in pars:
                        w.append(par)
                        
                    num = 0
                    for l_idx in range(len(w)):
                        num+= ((w[l_idx]-w0[l_idx])**2).mean()        
                    w_epochs.append((num/den).item())
                    print("Ev weights for P: "+str(P)+" and epoch: "+str(epoch)+" is: "+str((num/den).item()), flush = True) 
                      
                    #layer by layer
                    ev_layers = torch.zeros(nbl).to(device)

                    for mm in range(nbl):
                        ev_layers[mm] +=  ((net.conv[mm].w.data.clone() - net0.conv[mm].w.data.clone())**2).sum()
                        ev_layers[mm] +=  ((net.conv[mm].b.data.clone() - net0.conv[mm].b.data.clone())**2).sum()
                        ev_layers[mm] = ev_layers[mm]/ (((net0.conv[mm].w.data.clone())**2).sum() + ((net0.conv[mm].b.data.clone())**2).sum())

                    
                    #training loss
                    w_epochs_layers.append(ev_layers.cpu().detach().numpy())
                    print("Ev layers weights for P: "+str(P)+" and epoch: "+str(epoch)+" is: "+str(ev_layers), flush = True)


                    t = trLossEpoch/nt
                    t0 = (trLossWeightEpoch/nt).item()
                    print("Training error for P: "+str(P)+" and epoch: "+str(epoch)+" is: "+str(t), flush = True)
                    print("Training+Weight error for P: "+str(P)+" and epoch: "+str(epoch)+" is: "+str(t0), flush = True)
                    train_epochs.append(t)
                    w_reg.append(t0)
                    if t==0 and flag == 0:
                        print("ZERO train error for P: "+str(P)+" is "+str(t)+" at epoch: "+str(epoch), flush = True)
                        epoch_interpol = epoch
                        flag = 1
                    if ridge>0:
                        if epoch >= 500*epoch_interpol and t==0 and vali ==0:
                        
                            print("FINAL train error for P: "+str(P)+" is "+str(t), flush = True)
                            torch.save(net.state_dict(),r"models_save_1d/gap_P_%d_xi_%d_nbl_%d_aTilde_%d_LR_%.6f_epoch_%d_gap_%.1f_iter_%d_H_%d_magn_%.1f_rdg_%.6f_pv_%d_str_%d.pt"%(P,xi,nbl,A_exp,LR,epoch,gap,qq,H,magn,ridge,pv,stride))
                            break
                    else:
                        if t==0:
                            print("FINAL train error for P: "+str(P)+" is "+str(t), flush = True)
                            torch.save(net.state_dict(),r"models_save_1d/gap_P_%d_xi_%d_nbl_%d_aTilde_%d_LR_%.6f_epoch_%d_gap_%.1f_iter_%d_H_%d_magn_%.1f_rdg_%.6f_pv_%d_str_%d.pt"%(P,xi,nbl,A_exp,LR,epoch,gap,qq,H,magn,ridge,pv,stride))
                            break
                    
            
            if epoch==max_epochs-1:
                torch.save(net.state_dict(),r"models_save_1d/gap_P_%d_xi_%d_nbl_%d_aTilde_%d_LR_%.6f_epoch_%d_gap_%.1f_iter_%d_H_%d_magn_%.1f_rdg_%.6f_pv_%d_str_%d.pt"%(P,xi,nbl,A_exp,LR,epoch,gap,qq,H,magn,ridge,pv,stride))       
            epoch+=1
     
        print('Finished Training for P: '+str(P), flush = True)
        with torch.no_grad():
            L=0
            batch_test = 8
            NB = int(nval/batch_test)
            for hh in range(NB):
                it = image_test[hh*batch_test : (hh+1)*batch_test,:,:]
                lt = lab_test[hh*batch_test : (hh+1)*batch_test]

                out = net(it)
                
                L+=Loss01(out.squeeze(), lt.squeeze())
            L = L/NB
        lcurve[qq-1] = L
        print("test error for P: "+str(P)+" and iter:" +str(qq)+"is "+str(lcurve[qq-1]), flush = True)
        np.savetxt("files_save_1d/valErr_epochs_P_%d_xi_%d_nbl_%d_aTilde_%d_LR_%.6f_gap_%.1f_iter_%d_H_%d_magn_%.1f_rdg_%.6f_pv_%d_str_%d.txt"%(P,xi,nbl,A_exp,LR,gap,qq,H,magn,ridge,pv,stride),test_epochs)
        np.savetxt("files_save_1d/trainErr_epochs_P_%d_xi_%d_nbl_%d_aTilde_%d_LR_%.6f_gap_%.1f_iter_%d_H_%d_magn_%.1f_rdg_%.6f_pv_%d_str_%d.txt"%(P,xi,nbl,A_exp,LR,gap,qq,H,magn,ridge,pv,stride),train_epochs)
        np.savetxt("files_save_1d/wReg_epochs_P_%d_xi_%d_nbl_%d_aTilde_%d_LR_%.6f_gap_%.1f_iter_%d_H_%d_magn_%.1f_rdg_%.6f_pv_%d_str_%d.txt"%(P,xi,nbl,A_exp,LR,gap,qq,H,magn,ridge,pv,stride),w_reg) #.cpu().numpy())


        np.savetxt("files_save_1d/evolWeights_epochs_P_%d_xi_%d_nbl_%d_aTilde_%d_LR_%.6f_gap_%.1f_iter_%d_H_%d_magn_%.1f_rdg_%.6f_pv_%d_str_%d.txt"%(P,xi,nbl,A_exp,LR,gap,qq,H,magn,ridge,pv,stride),w_epochs)
        np.savetxt("files_save_1d/evolWeights_layers_epochs_P_%d_xi_%d_nbl_%d_aTilde_%d_LR_%.6f_gap_%.1f_iter_%d_H_%d_magn_%.1f_rdg_%.6f_pv_%d_str_%d.txt"%(P,xi,nbl,A_exp,LR,gap,qq,H,magn,ridge,pv,stride),w_epochs_layers)        

        np.savetxt("files_save_1d/lcurve_epochs_P_%d_xi_%d_nbl_%d_aTilde_%d_LR_%.6f_gap_%.1f_H_%d_magn_%.1f_rdg_%.6f_pv_%d_str_%d.txt"%(P,xi,nbl,A_exp,LR,gap,H,magn,ridge,pv,stride),lcurve.cpu().numpy())
        qq+=1


