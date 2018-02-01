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
import SimpleITK as sitk
import os

origin = 'Data/BRAST/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/preprocessed/'
path = os.listdir(origin)

train_x = []
train_y = []
train_s = []
train_t = []
test_s = []
test_t = []
test_s_y = []
test_t_y = []

def make_data(flag = 0):
    real_data = []
    fake_data = []
    if flag == 0:
        a = 0
        b = 30
    elif flag == 1:
        a = 30
        b = 40
    elif flag == 2:
        a = 40
        b = 50
    elif flag == 3:
        a = 0
        b = 70
    else:
        print "wrong number"
        return 1
    
    for i in path[a: b]:
        real_data = []
        fake_data = []
        if i.startswith('.'):
            continue
        sub = origin + "/" + i
        subsub = os.listdir(sub)
        if len(subsub) != 5:
            continue
            
        flair = sitk.ReadImage(sub + "/" + subsub[0])
        flair = sitk.GetArrayFromImage(flair)
        t1 = sitk.ReadImage(sub + "/" + subsub[1])
        t1 = sitk.GetArrayFromImage(t1)
        t2 = sitk.ReadImage(sub + "/" + subsub[3])
        t2 = sitk.GetArrayFromImage(t2)  
        y = sitk.ReadImage(sub + "/" + subsub[4])
        y = sitk.GetArrayFromImage(y)
        
        flair = flair[67 : 86, 110 : 129, 110 : 129]
        t1 = t1[67 : 86, 110 : 129, 110 : 129]
        t2 = t2[67 : 86, 110 : 129, 110 : 129]
        y = y[72 : 81, 115 : 124, 115 : 124]
        
        real_data.append(t2)
        real_data.append(t1)
        fake_data.append(t2)
        fake_data.append(flair)
        
        if flag == 0:
            train_x.append(real_data)
            train_y.append(y)
        elif flag == 1:
            train_s.append(real_data)
        elif flag == 2:
            train_t.append(fake_data)
        else: 
            test_s.append(real_data)
            test_s_y.append(y)
            test_t.append(fake_data)
            test_t_y.append(y)
            
make_data(0)
make_data(1)
make_data(2)
make_data(3)


train_x = np.array(train_x)
train_y = np.array(train_y)
train_s = np.array(train_s)
train_t = np.array(train_t)
test_s = np.array(test_s)
test_t = np.array(test_t)
test_s_y = np.array(test_s_y)
test_t_y = np.array(test_t_y)

class Segmenter(nn.Module):
    def __init__(self):
        super(Segmenter, self).__init__()
        self.res_normal_1 = nn.Sequential(nn.Conv3d(2, 30, 3), 
                                             nn.Conv3d(30, 30, 3), 
                                             nn.Conv3d(30, 40, 3),
                                             nn.Conv3d(40, 40, 3))
        self.res_normal_2 = nn.Sequential(nn.Conv3d(40, 40, 3),
                                             nn.Conv3d(40, 40, 3))
        self.res_normal_3 = nn.Sequential(nn.Conv3d(40, 50, 3),
                                             nn.Conv3d(50, 50, 3))
        self.res_low_1 = nn.Sequential(nn.Conv3d(2, 30, 3), 
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
    loss = input.clamp(min = 0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def discriminator_loss(h_real, h_fake):
    h_real = h_real.view(h_real.size(0), -1)
    h_fake = h_fake.view(h_fake.size(0), -1)
    target_real = Variable(torch.ones_like(h_real.data), requires_grad = False)
    target_fake = Variable(torch.zeros_like(h_fake.data), requires_grad = False)
    loss = bce_loss(h_real, target_real) + bce_loss(h_fake, target_fake)
    return loss

def segmenter_loss(y_predict, y_true, h_real, h_fake, alpha):
    y_predict = y_predict.view(y_predict.size(0) * 9 * 9 *9, -1)
    y_true = y_true.contiguous().view(y_true.size(0) * 9 * 9 * 9, -1)
    y_true = y_true.float()
    loss1 = bce_loss(y_predict, y_true)
    loss2 = 0
    if alpha != 0:
        loss2 =  discriminator_loss(h_real, h_fake)
    loss = loss1 - alpha * loss2
    return loss

def just_segmenter_loss(y_predict, y_true):
    y_predict = y_predict.view(y_predict.size(0) * 9 * 9 *9, -1)
    y_true = y_true.contiguous().view(y_true.size(0) * 9 * 9 * 9, -1)
    y_true = y_true.float()
    loss = bce_loss(y_predict, y_true)
    return loss

def get_optimizer(model):
    optimizer = None
    optimizer = optim.SGD(model.parameters(), lr = 0.0001)
    return optimizer

def gan(D, S, D_solver, S_solver, discriminator_loss, segmenter_loss, show_every = 250, 
              batch_size=1, num_epochs = 100):

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
            elif epoch < 35:
                S_error = segmenter_loss(y_predict, y_true, h_real, h_fake, 0.05 * (epoch - 9) / (34 - 9))
            else:
                S_error = segmenter_loss(y_predict, y_true, h_real, h_fake, 0.05)
            S_error.backward(retain_graph = True)
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

test_s = torch.from_numpy(test_s)
test_t = torch.from_numpy(test_t)
test_s_y = torch.from_numpy(test_s_y)
test_t_y = torch.from_numpy(test_t_y)
test_s = Variable(test_s)
test_t = Variable(test_t)
test_s_y = Variable(test_s_y)
test_t_y = Variable(test_t_y)
test_s_high = to_high_res(test_s)
test_t_high = to_high_res(test_t)

def just_segmenter(S, S_solver, segmenter_loss, show_every = 250, 
              batch_size=1, num_epochs = 100):

    iter_count = 0
    for epoch in range(num_epochs):
        #for x, y, x_s, x_t in zip(train_x, train_y, train_s, train_t):
            #if len(x) != batch_size:
             #   continue
            x = torch.from_numpy(train_x)
            y = torch.from_numpy(train_y)
            x = Variable(x)
            y_true = Variable(y)
            x_high = to_high_res(x)
            y_predict,  _ = S(x_high, x)
            S_solver.zero_grad()
            S_error = just_segmenter_loss(y_predict, y_true)
            S_error.backward(retain_graph = True)
            S_solver.step()
          #  if (iter_count % show_every == 0):
            print('Iter: {}, S:{:.4}'.format(iter_count, S_error.data[0]))
            iter_count += 1

D_1 = discriminator()
S_1 = Segmenter()
D_solver_1 = get_optimizer(D_1)
S_solver_1 = get_optimizer(S_1)
gan(D_1, S_1, D_solver_1, S_solver_1, discriminator_loss, segmenter_loss)

test_s_predict, _ = S_1(test_s_high, test_s)
test_t_predict, _ = S_1(test_t_high, test_t)
print just_segmenter_loss(test_s_predict, test_s_y)
print just_segmenter_loss(test_t_predict, test_t_y)

S_2 = Segmenter()
S_solver_2 = get_optimizer(S_2)
just_segmenter(S_2, S_solver_2, segmenter_loss)

test_s_predict, _ = S_2(test_s_high, test_s)
test_t_predict, _ = S_2(test_t_high, test_t)
print just_segmenter_loss(test_s_predict, test_s_y)
print just_segmenter_loss(test_t_predict, test_t_y)
       
