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


train_x = []
nrrd_filename = 'PDDCA-1.4.1_part1/0522c0001/img.nrrd'
nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
train_x.append(nrrd_data[246: 265, 235: 254, 62: 81].reshape(1, 19, 19, 19))
nrrd_filename = 'PDDCA-1.4.1_part1/0522c0002/img.nrrd'
nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
train_x.append(nrrd_data[240: 259, 213: 232, 81: 100].reshape(1, 19, 19, 19))
nrrd_filename = 'PDDCA-1.4.1_part1/0522c0003/img.nrrd'
nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
train_x.append(nrrd_data[240: 259, 250: 269, 88: 107].reshape(1, 19, 19, 19))
nrrd_filename = 'PDDCA-1.4.1_part1/0522c0009/img.nrrd'
nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
train_x.append(nrrd_data[240: 259, 240: 259, 95: 114].reshape(1, 19, 19, 19))

train_y = []
nrrd_filename = 'PDDCA-1.4.1_part1/0522c0001/structures/BrainStem.nrrd'
nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
train_y.append(nrrd_data[251: 260, 240: 249, 67: 76].reshape(1, 9, 9, 9))

nrrd_filename = 'PDDCA-1.4.1_part1/0522c0002/structures/BrainStem.nrrd'
nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
train_y.append(nrrd_data[245: 254, 218: 227, 86: 95].reshape(1, 9, 9, 9))

nrrd_filename = 'PDDCA-1.4.1_part1/0522c0003/structures/BrainStem.nrrd'
nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
print nrrd_data.shape
train_y.append(nrrd_data[245: 254, 255: 264, 93: 102].reshape(1, 9, 9, 9))

nrrd_filename = 'PDDCA-1.4.1_part1/0522c0009/structures/BrainStem.nrrd'
nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
print nrrd_data.shape
train_y.append(nrrd_data[245: 254, 245: 254, 100: 109].reshape(1, 9, 9, 9))


train_s = []
nrrd_filename = 'PDDCA-1.4.1_part1/0522c0013/img.nrrd'
nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
train_s.append(nrrd_data[232: 251, 245: 264, 91: 110].reshape(1, 19, 19, 19))
nrrd_filename = 'PDDCA-1.4.1_part1/0522c0014/img.nrrd'
nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
train_s.append(nrrd_data[242: 261, 271: 290, 93: 112].reshape(1, 19, 19, 19))
nrrd_filename = 'PDDCA-1.4.1_part1/0522c0017/img.nrrd'
nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
train_s.append(nrrd_data[240: 259, 204: 223, 100: 119].reshape(1, 19, 19, 19))
nrrd_filename = 'PDDCA-1.4.1_part1/0522c0057/img.nrrd'
nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
train_s.append(nrrd_data[240: 259, 210: 229, 105: 124].reshape(1, 19, 19, 19))


train_t = []
nrrd_filename = 'PDDCA-1.4.1_part1/0522c0070/img.nrrd'  #Mandible
nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
train_t.append(nrrd_data[210: 229, 200: 219, 65: 84].reshape(1, 19, 19, 19))
nrrd_filename = 'PDDCA-1.4.1_part1/0522c0077/img.nrrd'
nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
train_t.append(nrrd_data[240: 259, 240: 259, 80: 99].reshape(1, 19, 19, 19))
nrrd_filename = 'PDDCA-1.4.1_part1/0522c0079/img.nrrd'
nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
train_t.append(nrrd_data[210: 229, 230: 249, 65: 84].reshape(1, 19, 19, 19))
nrrd_filename = 'PDDCA-1.4.1_part1/0522c0081/img.nrrd'
nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
train_t.append(nrrd_data[190: 209, 200: 219, 90: 109].reshape(1, 19, 19, 19))

train_x = np.array(train_x)
train_y = np.array(train_y)
train_s = np.array(train_s)
train_t = np.array(train_t)


class Segmenter(nn.Module):
    def __init__(self):
        super(Segmenter, self).__init__()
        self.res_normal_1 = nn.Sequential(nn.Conv3d(1, 30, 3), 
                                             nn.Conv3d(30, 30, 3), 
                                             nn.Conv3d(30, 40, 3),
                                             nn.Conv3d(40, 40, 3))
        self.res_normal_2 = nn.Sequential(nn.Conv3d(40, 40, 3),
                                             nn.Conv3d(40, 40, 3))
        self.res_normal_3 = nn.Sequential(nn.Conv3d(40, 50, 3),
                                             nn.Conv3d(50, 50, 3))
        self.res_low_1 = nn.Sequential(nn.Conv3d(1, 30, 3), 
                                             nn.Conv3d(30, 30, 3), 
                                             nn.Conv3d(30, 40, 3),
                                             nn.Conv3d(40, 40, 3))
        self.res_low_2 = nn.Sequential(nn.Conv3d(40, 40, 3),
                                             nn.Conv3d(40, 40, 3))
        self.res_low_3 = nn.Sequential(nn.Conv3d(40, 50, 3),
                                             nn.Conv3d(50, 50, 3))
        self.res_1 = nn.Sequential(nn.Conv3d(100, 150, 1),
                                     nn.Conv3d(150, 150, 1))
        self.res_2 = nn.Conv3d(150, 1, 1)
        
    def forward(self, x, y):
        mm = nn.Upsample(scale_factor = 2, mode='nearest')
        N = x.size(0)
        
        x_normal_1 = self.res_normal_1(x)
        x_normal_c1 = x_normal_1[:, :, 4: 13, 4: 13, 4: 13]
        
        x_normal_2 = self.res_normal_2(x_normal_1)
        x_normal_c2 = x_normal_2[:, :, 2: 11, 2: 11, 2: 11]
        
        x_normal_3 = self.res_normal_3(x_normal_2)
        m = nn.Upsample(scale_factor = 3, mode='nearest')
        x_low_1 = self.res_low_1(y)
        x_low_up_1 = mm(x_low_1)
        x_low_c1 = x_low_up_1[:, :, 7: 16 , 7: 16, 7: 16]
        x_low_2 = self.res_low_2(x_low_1)
        x_low_up_2 = mm(x_low_2)
        x_low_c2 = x_low_up_2[:, :, 2: 11, 2: 11, 2: 11]
        x_low_3 = self.res_low_3(x_low_2)
        x_low_c3 = m(x_low_3)
        x_low = m(x_low_3)  
        conc = torch.cat((x_normal_3, x_low), dim = 1)
        out1 = self.res_1(conc)
        concat = torch.cat((x_low_c1, x_low_c2, x_low_c3, x_normal_c1, x_normal_c2, x_normal_3, out1), dim = 1)
        out = self.res_2(out1)
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


def segmenter_loss(y_predict, y_true, h_real, h_fake, alpha):
    y_predict = y_predict.view(y_predict.size(0) * 1 * 9 * 9 *9, -1)
    y_true = y_true.contiguous().view(y_true.size(0) * 1 * 9 * 9 * 9, -1)
    y_true = y_true.float()
    loss1 = bce_loss(y_predict, y_true)
    loss2 =  discriminator_loss(h_real, h_fake)
    loss = loss1 - alpha * loss2
    return loss

def get_optimizer(model):
   
    optimizer = None
    optimizer = optim.SGD(model.parameters(), lr = 0.001)
    return optimizer


def gan(D, S, D_solver, S_solver, discriminator_loss, segmenter_loss, show_every = 250, 
              batch_size=1, num_epochs=5):

    iter_count = 0

    for epoch in range(num_epochs):
        
        #for x, y, x_s, x_t in zip(train_x, train_y, train_s, train_t):
            #if len(x) != batch_size:
             #   continue
            
            x = torch.from_numpy(train_x)
            y = torch.from_numpy(train_y)
            x_s = torch.from_numpy(train_s)
            x_t = torch.from_numpy(train_t)
                      
            x = Variable(x)
            y_true = Variable(y)
            x_s = Variable(x_s)
            x_t = Variable(x_t)
            
            x_high = to_high_res(x)
            x_s_high = to_high_res(x_s)
            x_t_high = to_high_res(x_t)

            y_predict,  _ = S(x_high, x)
            _, out_real = S(x_s_high, x_s)
            _, out_fake = S(x_t_high, x_t)
            
            h_real = D(out_real)
            h_fake = D(out_fake)
            
            S_solver.zero_grad()
            if epoch < 10:
                S_error = segmenter_loss(y_predict, y_true, h_real, h_fake, 0)
                S_error.backward(retain_graph=True)
                S_solver.step()
            elif epoch < 35:
                S_error = segmenter_loss(y_predict, y_true, h_real, h_fake, 0)#0.05 * (epoch - 9) / (34 - 9))
                S_error.backward(retain_graph=True)
                S_solver.step()
            else:
                S_error = segmenter_loss(y_predict, y_true, h_real, h_fake, 0)#.05)
                S_error.backward(retain_graph=True)
                S_solver.step()
                
            D_solver.zero_grad()
            D_error = discriminator_loss(h_real, h_fake)
            D_error.backward()
            D_solver.step()

          #  if (iter_count % show_every == 0):
            print('Iter: {}, D: {:.4}, S:{:.4}'.format(iter_count, D_error.data[0], S_error.data[0]))
               
            iter_count += 1
            
def to_high_res(x):
    out = x[:, :, 3:16, 3:16, 3:16]
    mm = nn.Upsample(scale_factor = 2, mode = 'nearest')
    out = mm(out)[:, :, : 25, : 25, : 25]
    return out


D = discriminator()
S = Segmenter()

D_solver = get_optimizer(D)
S_solver = get_optimizer(S)

gan(D, S, D_solver, S_solver, discriminator_loss, segmenter_loss)



