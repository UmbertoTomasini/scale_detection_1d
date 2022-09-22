import torch
import torchvision
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from convolutional_nets import*

def from_coord_to_img (pts, img_size,magn):
        img = torch.zeros((img_size))
        img[int(pts[0,0])]=(magn)**.5
        img[int(pts[1,0])]=(magn)**.5
        return img
    
def over_boundaries(tdata_file,img_size):
    for jj in range(tdata_file.size(0)):
        if tdata_file[jj,0,0,0]== -1:
            tdata_file[jj,0,0,0] = 0 #img_size -1
        elif tdata_file[jj,0,0,0]== img_size:
            tdata_file[jj,0,0,0] = img_size -1 #0
            
            
        if tdata_file[jj,0,1,0]== -1:
            tdata_file[jj,0,1,0] =  0 #img_size -1
        elif tdata_file[jj,0,1,0]== img_size:
            tdata_file[jj,0,1,0] = img_size -1 #0
            

    return tdata_file
def perturb(imgs, timgs):
    """
    :param imgs: original images
    :param timgs: locally translated images (~diffeo)
    :return: original images, locally translated images, noisy images
    """
    sigma_sq = (timgs - imgs).pow(2).mean(2).mean(0)

    eta = torch.randn(imgs.shape)*(sigma_sq)**.5   #((8/3)*(img_size/2)/(img_size))**.5
    m = nn.ReLU()
    eta = m(eta)
    nimgs = imgs + eta
    return imgs, timgs, nimgs

def stability(f, i, ns):
    """
    compute stability of the function `f` to perturbations `ns` of `i`
    :param f: network function
    :param i: original image(s)
    :param ns: tensor of perturbed batches of images
    :return: stabilities
    """
    with torch.no_grad():
        f0 = f(i).detach().reshape(len(i), -1)  # [batch, ...]
        #print(f0.mean())
        deno = torch.cdist(f0, f0).pow(2).mean().item() + 1e-30
        #print(deno)
        
        S = []
        numS = []
        for n in ns:
            fn = f(n).detach().reshape(len(i), -1)  # [batch, ...]
            
            S += [
                (fn - f0).pow(2).mean(0).sum().item() / deno
            ]
            numS += [
                (fn - f0).pow(2).mean(0).sum().item() 
            ]
        return torch.tensor(S) , torch.tensor(numS), deno
    
def adding_diffeo(tdata_file, data_file ,xi):
    
    for k in range(data_file.size(0)):
        r1_prev = torch.div(data_file[k,0,0,0], xi, rounding_mode='floor') 
        r1_post = torch.div(tdata_file[k,0,0,0], xi, rounding_mode='floor') 
        if r1_post!= r1_prev:
            tdata_file[k,0,0,0] = data_file[k,0,0,0]

        r2_prev = torch.div(data_file[k,0,1,0], xi, rounding_mode='floor') 
        r2_post = torch.div(tdata_file[k,0,1,0], xi, rounding_mode='floor') 
        
        if r2_post!= r2_prev:
            tdata_file[k,0,1,0] = data_file[k,0,1,0]  
    return tdata_file
    


def stability_int(f, layer, i, ns, Mean = False):
    """
    compute stability of the function `f` to perturbations `ns` of `i`
    :param f: network function
    :param i: original image(s)
    :param ns: tensor of perturbed batches of images
    :return: stabilities
    """
    with torch.no_grad():
         
        if Mean == True:
            results = f(i)['conv.%d.mul'%(layer)].mean(1).unsqueeze(1)
        else:
            results = f(i)['conv.%d.mul'%(layer)] 
        f0 = results.detach().reshape(len(i), -1)  # [batch, ...]
        deno = torch.cdist(f0, f0).pow(2).mean().item() + 1e-30
        
        S = []
        numS = []
        for n in ns:
            if Mean == True:              
                results = f(n)['conv.%d.mul'%(layer)].mean(1).unsqueeze(1)
            else:
                results = f(n)['conv.%d.mul'%(layer)]
            norms = results.pow(2).sum([0,2]).pow(0.5)

            fn = results.detach().reshape(len(i), -1)  # [batch, ...]
            tmp0 = (fn - f0).pow(2).mean(0).sum().item()
            S += [
                 tmp0/ deno
            ]

        return torch.tensor(S) ,  deno



P = 1024
nbls = np.array([7])
a_exp = 0
a = 10.**(-a_exp)
lr = 0.01
xi = 32
gap = 0.
H=1000
img_size = 2**7
magn =img_size/2
ridges = np.array([0.01])
pv = 2
stride = 2
fsize = 2
idx_strides = [i for i in range(nbl)]

#choose whether mean channel or not
Mean =True
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

NTOT = int((nplus+nminus)/10)
data_file =torch.zeros((NTOT,1,2,1))
data_file[:int(NTOT/2),0,:,:] = data_file_plus[:int(NTOT/2),:,:]
data_file[int(NTOT/2):,0,:,:] = data_file_minus[:int(NTOT/2),:,:]


X_tr = torch.zeros([NTOT,1,img_size])
for gg in range(NTOT):
    X_tr[gg,0,:] = from_coord_to_img (data_file[gg,0,:], img_size,magn)
    
    
for rr in ridges:
    
    
    
    for nbl in nbls:
        
        
        fin_ep =2
        epochs = torch.tensor([1,fin_ep])
        
        ridge = rr
               
        ce = 0
        for epoch in epochs:
            cn=0
            for ll in range(nbl):
                print(ll)
                lctmp =np.loadtxt("files_save_1d/lcurve_epochs_P_%d_xi_%d_nbl_%d_aTilde_%d_LR_%.6f_gap_%.1f_H_%d_magn_%.1f_rdg_%.6f_pv_%d_str_%d.txt" %(P,xi,nbl,a_exp,lr,gap,H,m,ridge,pv,stride))
                num_nets = []
                for bb in range(len(lctmp)):
                    if lctmp[bb] <1:
                        num_nets.append(bb+1)
                
                
                rfs = torch.zeros((int(nbl),2,len(num_nets)))
                dfs = torch.zeros ((int(nbl),2,len(num_nets)))
                gfs = torch.zeros ((int(nbl),2,len(num_nets)))
                denos = torch.zeros ((int(nbl),2,len(num_nets)))
        

                cq = 0
                for qq in num_nets:

                    model = ConvNetGAPMF_1d(alpha_tilde = a, n_blocks= nbl , input_ch=1, h=H,  idx_strides = idx_strides, filter_size=fsize, stride_int=stride, out_dim=1)
                    if epoch>1:
                        traintmp=np.loadtxt("files_save_1d/trainErr_epochs_P_%d_xi_%d_nbl_%d_aTilde_%d_LR_%.6f_gap_%.1f_iter_%d_H_%d_magn_%.1f_rdg_%.6f_pv_%d_str_%d.txt" %(P,xi,nbl,a_exp,lr,gap,qq,H,m,ridge,pv,stride))
                        ee = (len(traintmp)-1)*10 +1
                        
                        model.load_state_dict(torch.load("models_save_1d/gap_P_%d_xi_%d_nbl_%d_aTilde_%d_LR_%.6f_epoch_%d_gap_%.1f_iter_%d_H_%d_magn_%.1f_rdg_%.6f_pv_%d_str_%d.pt"%(P,xi,nbl,a_exp,lr,ee-1,gap,qq,H,m,ridge,pv,stride), map_location=torch.device('cpu')))
                       
                    else:
                        ee = epoch
                        model.load_state_dict(torch.load("models_save_1d/gap_P_%d_xi_%d_nbl_%d_aTilde_%d_LR_%.6f_epoch_%d_gap_%.1f_iter_%d_H_%d_magn_%.1f_rdg_%.6f_pv_%d_str_%d.pt"%(P,xi,nbl,a_exp,lr,ee-1,gap,qq,H,m,ridge,pv,stride), map_location=torch.device('cpu')))
                    #diffeo
                    lt = 2*(torch.randint(2,data_file.size()) -1/2)
                    tdata_file = data_file + lt
                    tdata_file = over_boundaries(tdata_file,img_size)
                    tdata_file = adding_diffeo(tdata_file, data_file , xi)
                    
                    #correcting over the boundaries
                    tdata_file = over_boundaries(tdata_file,img_size)

                    tX_tr = torch.zeros([NTOT,1,img_size])
                    for hh in range(NTOT):

                        tX_tr[hh,0,:] = from_coord_to_img (tdata_file[hh,0,:,:], img_size)

                    X_tr, tX_tr, gX_tr = perturb(X_tr, tX_tr)

                    pX_tr = torch.zeros([2,NTOT,1,img_size])
                    pX_tr[0,:,:,:] = tX_tr
                    pX_tr[1,:,:,:] = gX_tr
                    
                    layers, _ = get_graph_node_names(model)
                    features = create_feature_extractor(model, return_nodes=layers)
                    
                    S, deno = stability_int(features,ll, X_tr, pX_tr,Mean)


                    rfs[cn,ce,cq] += S[0]/S[1]
                    dfs[cn,ce,cq] += S[0]
                    gfs[cn,ce,cq] += S[1]
                    denos[cn,ce,cq] += deno
                    
                    cq+=1


                cn+=1
                
            ce+=1
        rfs = rfs.numpy()
        dfs = dfs.numpy()
        gfs = gfs.numpy()
        denos = denos.numpy()

        ccq = 0
        for qq in num_nets:
            if Mean ==True:
                txt = "_mean_"
            else:
                txt = "_"
            np.savetxt("rf"+txt+"INT_k_%d_P_%d_xi_%d_gap_%.1f_a_%d_lr_%.6f_rdg_%.6f_pv_%d_str_%d_qq_%d.txt"%(nbl,P,xi,gap,a_exp,lr,ridge, pv,stride,qq),rfs[:,:,ccq])
            np.savetxt("df"+txt+"INT_k_%d_P_%d_xi_%d_gap_%.1f_a_%d_lr_%.6f_rdg_%.6f_pv_%d_str_%d_qq_%d.txt"%(nbl,P,xi,gap,a_exp,lr,ridge, pv,stride,qq),dfs[:,:,ccq])
            np.savetxt("gf"+txt+"INT_k_%d_P_%d_xi_%d_gap_%.1f_a_%d_lr_%.6f_rdg_%.6f_pv_%d_str_%d_qq_%d.txt"%(nbl,P,xi,gap,a_exp,lr,ridge, pv,stride,qq),gfs[:,:,ccq])
            np.savetxt("deno"+txt+"INT_k_%d_P_%d_xi_%d_gap_%.1f_a_%d_lr_%.6f_rdg_%.6f_pv_%d_str_%d_qq_%d.txt"%(nbl,P,xi,gap,a_exp,lr,ridge, pv,stride,qq),denos[:,:,ccq])

