#%%
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import torch.nn.functional as F
import scipy.io as sio
from pathlib import Path

num_bvalue = 15
num_data1 = 10
num_data2 = 10
b_value = [0, 20,40,60,80,100,200,400,600,800,1000,1300,1500,2000,2400]

#%%
# Final data
train_angle = 40;   para_sh_x= 0.0;         para_sc_x= 0.0;         para_tr_x= 10;          para_sh_y= 0.0;         para_sc_y= 0.0;         para_tr_y= 10
test_angle = 40;    test_para_sh_x= 0.0;    test_para_sc_x= 0.0;    test_para_tr_x= 10;     test_para_sh_y= 0.0;    test_para_sc_y= 0.0;    test_para_tr_y= 10

# --------------------Brain Input--------------------#
# Generate moved simulation data

#%%
def my_motion_no_sh(angle, tr_x, tr_y, S):  # angle:0~90 degree, tr_x = 0~1, tr_y = 0~1, S=np.array
    h=np.shape(S)[0]
    w=np.shape(S)[1]

    # rot_theta = np.radians(45)
    # tr_x = 20/128
    # tr_y = 10/128
    rot_theta = np.radians(angle)
    c, s = np.cos(rot_theta), np.sin(rot_theta)
    R = np.array(((c, -s, tr_x), (s, c, tr_y)))
    S_tmp = torch.tensor(S.astype(np.float32))
    theta_simul = torch.tensor(R, dtype=torch.float)
    theta_simul = theta_simul.view(-1, 2, 3)
    S_motion_slice = S_tmp.view(-1, 1, h, w)
    grid = F.affine_grid(theta_simul, S_motion_slice.size())
    out = F.grid_sample(S_motion_slice, grid, padding_mode='reflection')

    return out, theta_simul

# --------------------Brain Input--------------------#
# Generate moved simulation data
def my_motion(angle, tr_x, tr_y, sh_x, sh_y, sc_x, sc_y, S):  # angle:0~90 degree, tr_x = 0~1, tr_y = 0~1, S=np.array

    h=np.shape(S)[0]
    w=np.shape(S)[1]

    rot_theta = np.radians(angle)
    c, s = np.cos(rot_theta), np.sin(rot_theta)
    Sc = np.array(((sc_x,0),(0,sc_y)))

    R = np.array(((c, -s), (s, c)))
    Sh = np.array(((1, sh_x), (sh_y, 1)))
    ScSh = np.matmul(Sc,Sh)
    RScSh = np.matmul(R,ScSh)
    TRScSh = np.array(((RScSh[0][0], RScSh[0][1], tr_x),(RScSh[1][0], RScSh[1][1], tr_y)))

    S_tmp = torch.tensor(S.astype(np.float32))
    theta_simul = torch.tensor(TRScSh, dtype=torch.float)
    theta_simul = theta_simul.view(-1, 2, 3)
    S_motion_slice = S_tmp.view(-1, 1, h, w)
    grid = F.affine_grid(theta_simul, S_motion_slice.size())
    out = F.grid_sample(S_motion_slice, grid, padding_mode='reflection')
    return out
#%%
DWI_sig=np.flipud(sio.loadmat('S.mat')['S'])
DWI_mask=DWI_sig[:,:,0]>0

#%%
#---------------------Generate Training motion invivo data----------------------
h=np.shape(DWI_sig)[0]
w=np.shape(DWI_sig)[1]

DWI_motion = torch.zeros([h, w, num_bvalue])
DWI_original = torch.zeros([h, w, num_bvalue])
mask_motion = np.zeros([h, w, num_bvalue])

for data_i in range(100):
    for i in range(num_bvalue):
            if i==0:
                mask_motion[:, :, i] = DWI_mask.astype(bool)
                DWI_motion[:, :, i,] =  torch.tensor(DWI_sig[:,:,0].astype(np.float32))
                DWI_original[:, :, i] = torch.tensor(DWI_sig[:,:,0].astype(np.float32))
            else:
                train_angle_s = (random.random() - 0.5) * train_angle
                tr_x = (random.random() - 0.5) * para_tr_x / 128  #230mm/128*20
                tr_y = (random.random() - 0.5) * para_tr_y / 128
                sh_x = (random.random() - 0.5) * para_sh_x
                sh_y = (random.random() - 0.5) * para_sh_y
                sc_x = 1 + (random.random() - 0.5) * para_sc_x
                sc_y = 1 + (random.random() - 0.5) * para_sc_y

                DWI_motion[:,:,i] = my_motion(train_angle_s, tr_x, tr_y, sh_x, sh_y, sc_x, sc_y, DWI_sig[:,:,i])
                mask_motion[:,:,i] = my_motion(train_angle_s, tr_x, tr_y, sh_x, sh_y, sc_x, sc_y, DWI_mask.astype(bool))
                # #GROUP FIXED
                DWI_original[:,:,i] = torch.tensor(DWI_sig[:,:,i].astype(np.float32))

    #%%
    #----------------------------training data save-------------------------------------
    save_name ='DWIs_motion'+str(data_i)+'.mat'
    sio.savemat(save_name, {'DWI_motion': DWI_motion.detach().cpu().numpy(),
                            'mask_motion': mask_motion,
                            'DWI_original': np.array(DWI_original,dtype=float)
                            })
#%%
plt.figure(figsize=(16,10))
plt.xticks([])
plt.yticks([])
plt.box(None)
for i in range(15):
    plt.subplot(3,5,i+1)
    plt.title('b-value='+str(b_value[i])+' \n s/mm$^{2}$')
    plt.imshow(S_invivo_train[:,:,i],vmin=0,vmax=1000,cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()
# %%

