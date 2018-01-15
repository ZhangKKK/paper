import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



class Segmenter(nn.Module):
    def __init__(self):
        super(Segmenter, self).__init__()
        self.res_normal_1 = nn.Sequential(nn.Conv3d(4, 30, 3), 
                                             nn.Conv3d(30, 30, 3), 
                                             nn.Conv3d(30, 40, 3),
                                             nn.Conv3d(40, 40, 3))
        self.res_normal_2 = nn.Sequential(nn.Conv3d(40, 40, 3),
                                             nn.Conv3d(40, 40, 3))
        self.res_normal_3 = nn.Sequential(nn.Conv3d(40, 50, 3),
                                             nn.Conv3d(50, 50, 3))
        self.res_low_1 = nn.Sequential(nn.Conv3d(4, 30, 3), 
                                             nn.Conv3d(30, 30, 3), 
                                             nn.Conv3d(30, 40, 3),
                                             nn.Conv3d(40, 40, 3))
        self.res_low_2 = nn.Sequential(nn.Conv3d(40, 40, 3),
                                             nn.Conv3d(40, 40, 3))
        self.res_low_3 = nn.Sequential(nn.Conv3d(40, 50, 3),
                                             nn.Conv3d(50, 50, 3))
        self.fc_path_1 = nn.Sequential(nn.Linear(100 * 9 * 9 * 9, 150 * 9 * 9 * 9),
                                     nn.Linear(150 * 9 * 9 * 9, 150 * 9 * 9 * 9))
        self.fc_path_2 = nn.Linear(150 * 9 * 9 * 9, 2 * 9 * 9 * 9)
        
    def forward(self, x, y, alpha):
        
        mm = nn.Upsample(scale_factor = 2, mode='nearest')
        N = x.size(0)
        x_normal_1 = self.res_normal_1(x)
        x_normal_c1 = x_normal_1[:, :9, :9, :9]
        x_normal_2 = self.res_normal_2(x_normal_1)
        x_normal_c2 = x_normal_2[:, :9, :9, :9]
        x_normal_3 = self.res_normal_3(x_normal_2)
        m = nn.Upsample(scale_factor = 3, mode='nearest')
        x_low_1 = self.res_low_1(y)
        x_low_up_1 = mm(x_low_1)
        x_low_c1 = x_low_up_1[:, :9, :9, :9]
        x_low_2 = self.res_low_2(x_low_1)
        x_low_up_2 = mm(x_low_2)
        x_low_c2 = x_low_up_2[:, :9, :9, :9]
        x_low_3 = self.res_low_3(x_low_2)
        x_low_c3 = m(x_low_3)
        
      
        
        x_low = m(x_low_3)  
        
        conc = torch.cat(x_normal_3, x_low)
        N = conc.size(0) 
        out_1 = out.view(N, -1)       
        out_2 = self.fc_path_1(out_1)
        concat = torch.cat(x_low_c1, x_low_c2, x_low_c3, x_normal_c1, x_normal_c2, x_normal_3, out_2)
        out_3 = self.fc_path_2(out_2)
        out = out_3.view(N, 2, 9, 9, 9)
        
        return out, concat



def discriminator():
    
    model = nn.Sequential(
        nn.Conv3d(410, 100, 3),
        nn.Conv3d(100, 100, 3),
        nn.Conv3d(100, 100, 3),
        nn.Conv3d(100, 100, 3),
        nn.Conv3d(100, 1, 1)
        
    )
    return model


def bce_loss(input, target):
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()
   

def discriminator_loss(h_real, h_fake):
    target_real = Variable(torch.ones_like(h_real.data), requires_grad = False)
    target_fake = Variable(torch.zeros_like(h_fake.data), requires_grad = False)
    loss = bce_loss(h_real, target_real) + bce_loss(h_fake, target_fake)

    return loss


def segmenter_loss(y_predict, y_true, alpha, h_real, h_fake, alpha):
    y_predit = y_predict.view(y_predict.size(0) * 2 * 9 * 9 *9, -1)
    y_true = y_true.view(y_true.size(0) * 2 * 9 * 9 * 9, -1)
    loss1 = bce_loss(y_predict, y_true)
    
    
    loss2 =  discriminator_loss(h_real, h_fake)

    loss = loss1 - alpha * loss2
    return loss


def get_optimizer(model):
   
    optimizer = None
    optimizer = optim.SGD(model.parameters(), lr = 0.001)
    return optimizer








def gan(D, S, D_solver, S_solver, discriminator_loss, segmenter_loss, show_every=250, 
              batch_size=128, num_epochs=50):

    iter_count = 0
    for epoch in range(num_epochs):
        
        for x, y, x_s, x_t in loader_train:
            if len(x) != batch_size:
                continue
            
            x = Variable(x)
            y = Variable(Y)
            x_s = Variable(x_s)
            x_t = Variable(x_t)
            
            x_high = to_high_res(x)
            x_s_high = to_high_res(x_s)
            x_t_high = to_high_res(x_t)

            x_high = Variable(x_high)
            x_s_high = Variable(x_s_high)
            x_t_high = Variable(x_t_high)
            
            D_solver.zero_grad()
            
            
            y_predict,  _ = S(x_high, x)
            _, out_real = S(x_s_high, x_s)
            _, out_fake = S(x_t_high, x_t)
            
            
            h_real = D(out_real)
            h_fake = D(out_fake)
            
            D_error = discriminator_loss(h_real, h_fake)
            
            D_error.backward()
            D_solver.step()
            
            h_real = D(out_real)
            h_fake = D(out_fake)
            
            S_solver.zero_grad()
            
            if epoch < 10:
                
                S_error = segmenter_loss(y_predict, y_true, alpha, h_real, h_fake, 0)
                S_error.backward()
                S_solver.step()
                
            else if epoch < 35:
                S_error = segmenter_loss(y_predict, y_true, alpha, h_real, h_fake, 0.05 * (epoch - 9) / (34 - 9))
                S_error.backward()
                S_solver.step()
                
            else:
                S_error = segmenter_loss(y_predict, y_true, alpha, h_real, h_fake, 0.05)
                S_error.backward()
                S_solver.step()
            
            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, D_error.data[0], G_error.data[0]))
               
            iter_count += 1
            
def to_high_res(x):
    out = x[:, 3:15, 3:15, 3:15]
    mm = nn.Upsample(scale_factor = 2, mode='nearest')
    out = mm(out)[:, :25, :25, :25]
    return out
