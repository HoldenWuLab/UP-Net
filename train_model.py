import os
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import random

from models.model_UP_Net import UP_Net
from loss.generate_syn_images import generate_syn_images
from loss.loss_uncert import loss_uncert


def train(epoch):
    
    network.train()
    
    loss_one_epoch = []
    loss_img_epoch = []
    loss_map_epoch = []
    loss_enh_epoch = []
    loss_unc_epoch = []
    
    weight_cliping_limit = 0.01
    one = torch.FloatTensor([1])
    mone = one * -1
    one = one.to(device)
    mone = mone.to(device)
    phase_cycling = 1
    
    for i in range(batch_number):
    
        # To load data
        # img_undersamp, img_ref, map_ref = dataloader()
        Nec = 6
        
        random.seed(datetime.now())
        r_rotate = np.random.randint(0, 4)
        r_flip = np.random.randint(0, 2)
        
        if r_flip == 1:
            img_undersamp = np.flip(img_undersamp, axis=3)
            img_ref = np.flip(img_ref, axis=3)
            map_ref = np.flip(map_ref, axis=3)
        if r_rotate != 0:
            img_undersamp = np.rot90(img_undersamp, k = r_rotate, axes=(2,3))
            img_ref = np.rot90(img_ref, k = r_rotate, axes=(2,3))
            map_ref = np.rot90(map_ref, k = r_rotate, axes=(2,3))
        
        if phase_cycling == 1:
            random_theta = 2*np.pi*random.random()
            img_undersamp_phase = np.zeros_like(img_undersamp)
            img_ref_phase = np.zeros_like(img_ref)
            for cc in range(Nec):
                cs_phase[:,cc,:,:]    = cs_batch[:,cc,:,:]    * np.cos(random_theta) - cs_batch[:,cc+6,:,:]  * np.sin(random_theta)
                cs_phase[:,cc+6,:,:]  = cs_batch[:,cc+6,:,:]  * np.cos(random_theta) + cs_batch[:,cc,:,:]    * np.sin(random_theta)
                ini_phase[:,cc,:,:]   = ini_batch[:,cc,:,:]   * np.cos(random_theta) - ini_batch[:,cc+6,:,:] * np.sin(random_theta)
                ini_phase[:,cc+6,:,:] = ini_batch[:,cc+6,:,:] * np.cos(random_theta) + ini_batch[:,cc,:,:]   * np.sin(random_theta)
            
        img_undersamp = torch.from_numpy(img_undersamp.copy())
        img_undersamp = img_undersamp.float().to(device)
        img_ref = torch.from_numpy(img_ref.copy())
        img_ref = img_ref.float().to(device)
        map_ref = torch.from_numpy(map_ref.copy())
        map_ref = map_ref.float().to(device)

        optimizer.zero_grad()
        
        for p in netD.parameters():
            p.requires_grad = True
        
        optimizerD.zero_grad()
        for p in netD.parameters():
            p.data.clamp_(-1*weight_cliping_limit, weight_cliping_limit)
        
        d_loss_ref = netD(img_ref.detach())
        d_loss_ref = d_loss_ref.mean(0).view(1)
        d_loss_ref.backward()
        
        img_output, map_output, unc_output = network(img_undersamp)
        d_loss_gen = netD(img_output.detach())
        d_loss_gen = d_loss_gen.mean(0).view(1)
        d_loss_gen.backward(mone)
        
        loss_D = d_loss_gen - d_loss_ref
        optimizerD.step()
        
        for p in netD.parameters():
            p.requires_grad = False 
        
        output_map_w_sig = torch.sqrt(torch.pow(map_output[:,0,:,:],2) + torch.pow(map_output[:,1,:,:],2) + 1e-10)
        output_map_f_sig = torch.sqrt(torch.pow(map_output[:,2,:,:],2) + torch.pow(map_output[:,3,:,:],2) + 1e-10)
        output_map_ff = torch.div(output_map_f_sig, output_map_w_sig + output_map_f_sig+1e-10)
        output_map_mag = torch.stack((output_map_ff, map_output[:,4,:,:], map_output[:,5,:,:]),dim=1)
        
        syn_img = syn_func(map_output)
        
        loss_gan = netD(img_output)
        loss_gan = loss_gan.mean().mean(0).view(1)
        loss_enh = mseloss(img_output, img_ref)
        loss_map = mseloss(output_map_mag, map_ref)
        loss_img = mseloss(syn_img, img_ref)
        loss_unc = uncloss(output_map_mag, map_ref, unc_output)
        
        loss = 0.3*loss_enh + 0.1*loss_gan + 0.3*loss_map + 0.2*loss_img  + 0.1*loss_unc
        
        loss.backward()
        
        optimizer.step()

        loss_one_epoch.append(loss.item())
        loss_enh_epoch.append(loss_enh.item())
        loss_gan_epoch.append(loss_gan.item())
        loss_map_epoch.append(loss_map.item())
        loss_img_epoch.append(loss_img.item())
        loss_unc_epoch.append(loss_unc.item())
        
        
if __name__ == "__main__":

    network = UP_Net()
    netD = UP_Net_dis()
    network.load_state_dict(torch.load('/models/UP-Net_models.pt'), strict=False)
    netD.load_state_dict(torch.load('/models/UP-Net_models_dis.pt'), strict=False)

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    
    batch_size = 12
    num_epochs = 150
    lr_rate = 0.00001
    phase_cycling = 1

    network.cuda()
    netD.cuda()

    syn_func = generate_syn_images(device,batch_size)
    mseloss = nn.MSELoss()
    uncloss = loss_uncert()
    
    optimizer = optim.Adam(network.parameters(), lr=lr_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizerD = optim.Adam(netD.parameters(), lr=lr_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    for i in tqdm(range(num_epochs)):
        output = train(i)
    
    save_filename = 'UP_Net_results'
    torch.save(network.state_dict(),'/models/' + save_filename + '.pt')

    