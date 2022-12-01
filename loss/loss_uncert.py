import torch.nn as nn
import torch
import math

class loss_uncert(nn.Module):
    
    def __init__(self):
        super(loss_uncert,self).__init__()
        
        self.pi = math.pi
        self.eps = 1e-10
        
    def forward(self, output_img, ref_img, uncert_img):
        
        Nbatch, Nch, Nx, Ny = output_img.shape
        uncert_img = uncert_img.unsqueeze(1)

        uncert_loss = ( torch.sum( torch.div(torch.abs(output_img - ref_img), uncert_img + self.eps)/Nch ) + torch.sum(torch.log(uncert_img)) ) / (Nbatch*Nx*Ny)
                
        return uncert_loss
