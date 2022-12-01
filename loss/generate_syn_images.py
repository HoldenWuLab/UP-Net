import torch.nn as nn
import torch

class generate_syn_images(nn.Module):
    def __init__(self, device, batch_size):
        super(generate_syn_images, self).__init__()
        
        self.device = device
        self.pi = 2*torch.acos(torch.zeros(1)).item()
        self.fat_freq = torch.tensor([-3.72928, -3.32928, -3.03928, -2.59928, -2.37928, -1.85928, 0.680722])
        self.fat_amps = torch.tensor([0.0825134, 0.627312, 0.0715598, 0.0957628, 0.065591, 0.0153483, 0.0419126])
        self.fat_freq = self.fat_freq.float().to(device)
        self.fat_amps = self.fat_amps.float().to(device)
        self.TE = torch.tensor([1.23e-3, 2.46e-3, 3.69e-3, 4.92e-3, 6.15e-3, 7.38e-3])
        self.r2s_map_scale = 1/100
        self.field_map_scale = 1/100
        self.T = 2.8936
        self.batch_size = batch_size
        
    def forward(self, output_maps):
        
        bvec = torch.tensor([self.T]*self.batch_size).float().to(self.device)
        fat_freq_b0 = torch.ger(self.fat_freq, bvec) * 42.58
        Pr_ec0 = torch.matmul(self.fat_amps, torch.cos(2*self.pi*self.TE[0]*fat_freq_b0))
        Pi_ec0 = torch.matmul(self.fat_amps, torch.sin(2*self.pi*self.TE[0]*fat_freq_b0))
        Pr_ec1 = torch.matmul(self.fat_amps, torch.cos(2*self.pi*self.TE[1]*fat_freq_b0))
        Pi_ec1 = torch.matmul(self.fat_amps, torch.sin(2*self.pi*self.TE[1]*fat_freq_b0))
        Pr_ec2 = torch.matmul(self.fat_amps, torch.cos(2*self.pi*self.TE[2]*fat_freq_b0))
        Pi_ec2 = torch.matmul(self.fat_amps, torch.sin(2*self.pi*self.TE[2]*fat_freq_b0))
        Pr_ec3 = torch.matmul(self.fat_amps, torch.cos(2*self.pi*self.TE[3]*fat_freq_b0))
        Pi_ec3 = torch.matmul(self.fat_amps, torch.sin(2*self.pi*self.TE[3]*fat_freq_b0))
        Pr_ec4 = torch.matmul(self.fat_amps, torch.cos(2*self.pi*self.TE[4]*fat_freq_b0))
        Pi_ec4 = torch.matmul(self.fat_amps, torch.sin(2*self.pi*self.TE[4]*fat_freq_b0))
        Pr_ec5 = torch.matmul(self.fat_amps, torch.cos(2*self.pi*self.TE[5]*fat_freq_b0))
        Pi_ec5 = torch.matmul(self.fat_amps, torch.sin(2*self.pi*self.TE[5]*fat_freq_b0))
        
        r2s_ec0 = torch.exp(-1*output_maps[:,4,:,:]*self.TE[0]/self.r2s_map_scale)
        r2s_ec1 = torch.exp(-1*output_maps[:,4,:,:]*self.TE[1]/self.r2s_map_scale)
        r2s_ec2 = torch.exp(-1*output_maps[:,4,:,:]*self.TE[2]/self.r2s_map_scale)
        r2s_ec3 = torch.exp(-1*output_maps[:,4,:,:]*self.TE[3]/self.r2s_map_scale)
        r2s_ec4 = torch.exp(-1*output_maps[:,4,:,:]*self.TE[4]/self.r2s_map_scale)
        r2s_ec5 = torch.exp(-1*output_maps[:,4,:,:]*self.TE[5]/self.r2s_map_scale)
        r2s_ec0 = torch.transpose(r2s_ec0, 0,2)
        r2s_ec1 = torch.transpose(r2s_ec1, 0,2)
        r2s_ec2 = torch.transpose(r2s_ec2, 0,2)
        r2s_ec3 = torch.transpose(r2s_ec3, 0,2)
        r2s_ec4 = torch.transpose(r2s_ec4, 0,2)
        r2s_ec5 = torch.transpose(r2s_ec5, 0,2)
        
        fm_r_ec0 = torch.cos(2*self.pi*output_maps[:,5,:,:]*self.TE[0]/self.field_map_scale)
        fm_i_ec0 = torch.sin(2*self.pi*output_maps[:,5,:,:]*self.TE[0]/self.field_map_scale)
        fm_r_ec1 = torch.cos(2*self.pi*output_maps[:,5,:,:]*self.TE[1]/self.field_map_scale)
        fm_i_ec1 = torch.sin(2*self.pi*output_maps[:,5,:,:]*self.TE[1]/self.field_map_scale)
        fm_r_ec2 = torch.cos(2*self.pi*output_maps[:,5,:,:]*self.TE[2]/self.field_map_scale)
        fm_i_ec2 = torch.sin(2*self.pi*output_maps[:,5,:,:]*self.TE[2]/self.field_map_scale)
        fm_r_ec3 = torch.cos(2*self.pi*output_maps[:,5,:,:]*self.TE[3]/self.field_map_scale)
        fm_i_ec3 = torch.sin(2*self.pi*output_maps[:,5,:,:]*self.TE[3]/self.field_map_scale)
        fm_r_ec4 = torch.cos(2*self.pi*output_maps[:,5,:,:]*self.TE[4]/self.field_map_scale)
        fm_i_ec4 = torch.sin(2*self.pi*output_maps[:,5,:,:]*self.TE[4]/self.field_map_scale)
        fm_r_ec5 = torch.cos(2*self.pi*output_maps[:,5,:,:]*self.TE[5]/self.field_map_scale)
        fm_i_ec5 = torch.sin(2*self.pi*output_maps[:,5,:,:]*self.TE[5]/self.field_map_scale)
        fm_r_ec0 = torch.transpose(fm_r_ec0, 0,2)
        fm_i_ec0 = torch.transpose(fm_i_ec0, 0,2)
        fm_r_ec1 = torch.transpose(fm_r_ec1, 0,2)
        fm_i_ec1 = torch.transpose(fm_i_ec1, 0,2)
        fm_r_ec2 = torch.transpose(fm_r_ec2, 0,2)
        fm_i_ec2 = torch.transpose(fm_i_ec2, 0,2)
        fm_r_ec3 = torch.transpose(fm_r_ec3, 0,2)
        fm_i_ec3 = torch.transpose(fm_i_ec3, 0,2)
        fm_r_ec4 = torch.transpose(fm_r_ec4, 0,2)
        fm_i_ec4 = torch.transpose(fm_i_ec4, 0,2)
        fm_r_ec5 = torch.transpose(fm_r_ec5, 0,2)
        fm_i_ec5 = torch.transpose(fm_i_ec5, 0,2)

        wr = output_maps[:,0,:,:]
        wi = output_maps[:,1,:,:]
        fr = output_maps[:,2,:,:]
        fi = output_maps[:,3,:,:]
        wr = torch.transpose(wr, 0,2)
        wi = torch.transpose(wi, 0,2)
        fr = torch.transpose(fr, 0,2)
        fi = torch.transpose(fi, 0,2)
        
        ec0_syn_r = (torch.mul(wr + fr*Pr_ec0 - fi*Pi_ec0, fm_r_ec0) - torch.mul(wi + fr*Pi_ec0 + fi*Pr_ec0, fm_i_ec0))*r2s_ec0
        ec0_syn_i = (torch.mul(wi + fr*Pi_ec0 + fi*Pr_ec0, fm_r_ec0) + torch.mul(wr + fr*Pr_ec0 - fi*Pi_ec0, fm_i_ec0))*r2s_ec0
        ec1_syn_r = (torch.mul(wr + fr*Pr_ec1 - fi*Pi_ec1, fm_r_ec1) - torch.mul(wi + fr*Pi_ec1 + fi*Pr_ec1, fm_i_ec1))*r2s_ec1
        ec1_syn_i = (torch.mul(wi + fr*Pi_ec1 + fi*Pr_ec1, fm_r_ec1) + torch.mul(wr + fr*Pr_ec1 - fi*Pi_ec1, fm_i_ec1))*r2s_ec1
        ec2_syn_r = (torch.mul(wr + fr*Pr_ec2 - fi*Pi_ec2, fm_r_ec2) - torch.mul(wi + fr*Pi_ec2 + fi*Pr_ec2, fm_i_ec2))*r2s_ec2
        ec2_syn_i = (torch.mul(wi + fr*Pi_ec2 + fi*Pr_ec2, fm_r_ec2) + torch.mul(wr + fr*Pr_ec2 - fi*Pi_ec2, fm_i_ec2))*r2s_ec2
        ec3_syn_r = (torch.mul(wr + fr*Pr_ec3 - fi*Pi_ec3, fm_r_ec3) - torch.mul(wi + fr*Pi_ec3 + fi*Pr_ec3, fm_i_ec3))*r2s_ec3
        ec3_syn_i = (torch.mul(wi + fr*Pi_ec3 + fi*Pr_ec3, fm_r_ec3) + torch.mul(wr + fr*Pr_ec3 - fi*Pi_ec3, fm_i_ec3))*r2s_ec3
        ec4_syn_r = (torch.mul(wr + fr*Pr_ec4 - fi*Pi_ec4, fm_r_ec4) - torch.mul(wi + fr*Pi_ec4 + fi*Pr_ec4, fm_i_ec4))*r2s_ec4
        ec4_syn_i = (torch.mul(wi + fr*Pi_ec4 + fi*Pr_ec4, fm_r_ec4) + torch.mul(wr + fr*Pr_ec4 - fi*Pi_ec4, fm_i_ec4))*r2s_ec4
        ec5_syn_r = (torch.mul(wr + fr*Pr_ec5 - fi*Pi_ec5, fm_r_ec5) - torch.mul(wi + fr*Pi_ec5 + fi*Pr_ec5, fm_i_ec5))*r2s_ec5
        ec5_syn_i = (torch.mul(wi + fr*Pi_ec5 + fi*Pr_ec5, fm_r_ec5) + torch.mul(wr + fr*Pr_ec5 - fi*Pi_ec5, fm_i_ec5))*r2s_ec5
        
        ec0_syn_r = torch.transpose(ec0_syn_r, 0,2)
        ec0_syn_i = torch.transpose(ec0_syn_i, 0,2)
        ec1_syn_r = torch.transpose(ec1_syn_r, 0,2)
        ec1_syn_i = torch.transpose(ec1_syn_i, 0,2)
        ec2_syn_r = torch.transpose(ec2_syn_r, 0,2)
        ec2_syn_i = torch.transpose(ec2_syn_i, 0,2)
        ec3_syn_r = torch.transpose(ec3_syn_r, 0,2)
        ec3_syn_i = torch.transpose(ec3_syn_i, 0,2)
        ec4_syn_r = torch.transpose(ec4_syn_r, 0,2)
        ec4_syn_i = torch.transpose(ec4_syn_i, 0,2)
        ec5_syn_r = torch.transpose(ec5_syn_r, 0,2)
        ec5_syn_i = torch.transpose(ec5_syn_i, 0,2)
                
        output_ec = torch.stack([ec0_syn_r, ec1_syn_r, ec2_syn_r, ec3_syn_r, ec4_syn_r, ec5_syn_r, ec0_syn_i, ec1_syn_i, ec2_syn_i, ec3_syn_i, ec4_syn_i, ec5_syn_i], 1)
        
        
        return output_ec