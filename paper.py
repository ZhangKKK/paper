import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torchvision
import torch.utils.data as Data 
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import SimpleITK as sitk
import os

origin = 'Data/BRAST/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/preprocessed/'
path = os.listdir(origin)

train_x, train_y, train_s, train_t, test_s, test_t, test_s_y, test_t_y = \
                                    [], [], [], [], [], [], [], []

def make_data(flag = 0):
    if flag == 0:
        a = 0
        b = 60
    elif flag == 1:
        a = 60
        b = 70
    elif flag == 2:
        a = 0
        b = 60
    else:
        print "wrong number"
        return 1
    
    for i in path[a: b]:
        real_data, fake_data, y_data = [], [], []
        if i.startswith('.'):
            continue
        sub = origin + "/" + i
        subsub = os.listdir(sub)
        if len(subsub) != 5:
            continue
        curr = [0] * 5
        for j in range(5):
            curr[j] = sitk.ReadImage(sub + "/" + subsub[j])
            curr[j] = sitk.GetArrayFromImage(curr[j])
            
            if j < 4:
                curr[j] = curr[j][67 : 86, 110 : 129, 110 : 129]
                
            else:
                curr[j] = curr[j][72 : 81, 115 : 124, 115 : 124]
        flair, t1, t1gd, t2, y = curr
        y[y != 0] = 1
        
        y_data.append(y)
        real_data.append(flair)
        real_data.append(t2)
   #     real_data.append(t1)
        fake_data.append(flair)
        fake_data.append(t2)
    #    fake_data.append(t1gd)
        
        if flag == 0:
            train_x.append(real_data)
            train_y.append(y_data)
        elif flag == 1:
            train_s.append(real_data)
            train_t.append(fake_data)
        else: 
            test_s.append(real_data)
            test_s_y.append(y_data)
            test_t.append(fake_data)
            test_t_y.append(y_data)

make_data(0)
make_data(1)
make_data(2)

dtype = torch.cuda.FloatTensor

def to_high_res(x):
    out = x[:, :, 3:16, 3:16, 3:16]
    mm = nn.Upsample(scale_factor = 2, mode = 'nearest')
    out = mm(out)[:, :, : 25, : 25, : 25]
    return out

train_x = np.array(train_x)
train_y = np.array(train_y)
train_s = np.array(train_s)
train_t = np.array(train_t)
test_s = np.array(test_s)
test_t = np.array(test_t)
test_s_y = np.array(test_s_y)
test_t_y = np.array(test_t_y)
x_st = np.concatenate((train_s, train_t), axis = 0)
train_s_y = np.ones(10)
train_t_y = np.zeros(10)
y_st = np.concatenate((train_s_y, train_t_y), axis = 0)

x = torch.from_numpy(train_x)
y = torch.from_numpy(train_y)
x_st = torch.from_numpy(x_st)
y_st = torch.from_numpy(y_st)
test_s = torch.from_numpy(test_s)
test_t = torch.from_numpy(test_t)
test_s_y= torch.from_numpy(test_s_y)
test_t_y = torch.from_numpy(test_t_y)

test_s = Variable(test_s).type(dtype)
test_t = Variable(test_t).type(dtype)
test_s_y = Variable(test_s_y).type(dtype)
test_t_y = Variable(test_t_y).type(dtype)

test_s_high = to_high_res(test_s)
test_t_high = to_high_res(test_t)

class Segmenter(nn.Module):
    def __init__(self):
        super(Segmenter, self).__init__()
        self.res_normal_1 = nn.Sequential(nn.Conv3d(2, 30, 3), 
                                             nn.Conv3d(30, 30, 3), 
                                             nn.Conv3d(30, 40, 3),
                                             nn.Conv3d(40, 40, 3)).type(dtype)
        self.res_normal_2 = nn.Sequential(nn.Conv3d(40, 40, 3),
                                             nn.Conv3d(40, 40, 3)).type(dtype)
        self.res_normal_3 = nn.Sequential(nn.Conv3d(40, 50, 3),
                                             nn.Conv3d(50, 50, 3)).type(dtype)
        self.res_low_1 = nn.Sequential(nn.Conv3d(2, 30, 3), 
                                             nn.Conv3d(30, 30, 3), 
                                             nn.Conv3d(30, 40, 3),
                                             nn.Conv3d(40, 40, 3)).type(dtype)
        self.res_low_2 = nn.Sequential(nn.Conv3d(40, 40, 3),
                                             nn.Conv3d(40, 40, 3)).type(dtype)
        self.res_low_3 = nn.Sequential(nn.Conv3d(40, 50, 3),
                                             nn.Conv3d(50, 50, 3)).type(dtype)
        self.res_1 = nn.Sequential(nn.Conv3d(100, 150, 1),
                                     nn.Conv3d(150, 150, 1)).type(dtype)
        self.res_2 = nn.Conv3d(150, 1, 1).type(dtype)
        
    def forward(self, x, y):
        mm = nn.Upsample(scale_factor = 2, mode = 'nearest')
        N = x.size(0)
        
        x_normal_1 = self.res_normal_1(x)
        x_normal_c1 = x_normal_1[:, :, 4: 13, 4: 13, 4: 13]
        
        x_normal_2 = self.res_normal_2(x_normal_1)
        x_normal_c2 = x_normal_2[:, :, 2: 11, 2: 11, 2: 11]
        
        x_normal_3 = self.res_normal_3(x_normal_2)
        m = nn.Upsample(scale_factor = 3, mode = 'nearest')
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
     ).type(dtype)
    return model

def bce_loss(input, target):
    neg_abs = - input.abs()
    loss = input.clamp(min = 0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def discriminator_loss(y_st_predict, y_st):
    loss = bce_loss(y_st_predict, y_st)
    return loss

def segmenter_loss(y_predict, y_true, y_st_predict, y_st, alpha):
    y_predict = y_predict.view(y_predict.size(0) * 9 * 9 * 9, -1)
    y_true = y_true.contiguous().view(y_true.size(0) * 9 * 9 * 9, -1)
    y_true = y_true.float()
    loss1 = bce_loss(y_predict, y_true)
    loss2 = 0
    if alpha != 0:
        loss2 =  discriminator_loss(y_st_predict, y_st)
    loss = loss1 - alpha * loss2
    return loss

def just_segmenter_loss(y_predict, y_true):
    y_predict = y_predict.view(y_predict.size(0) * 9 * 9 * 9, -1)
    y_true = y_true.contiguous().view(y_true.size(0) * 9 * 9 * 9, -1)
    y_true = y_true.float()
    loss = bce_loss(y_predict, y_true)
    return loss

def compute_dsc_precision_recall(y_predict, y_true, thereshold):
    y_predict = torch.sigmoid(y_predict)
    y_predict = (y_predict >= thereshold)
    print float((y_predict.data.byte() == y_true.data.byte()).sum()) / (y_predict.size(0) * 9 * 9 * 9)
    print float(((y_predict.data.byte() == y_true.data.byte()) & (y_predict.data == 1)).sum()) / torch.sum(y_predict.data == 1)
    print float(((y_predict.data.byte() == y_true.data.byte()) & (y_predict.data == 1)).sum()) / torch.sum(y_true.data == 1)

def get_optimizer(model):
    optimizer = None
    optimizer = optim.SGD(model.parameters(), lr = 0.0001)
    return optimizer

class TensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

tensor_dataset = TensorDataset(x, y)
tensor_dataloader = DataLoader(tensor_dataset, batch_size = 5, shuffle = True, num_workers = 0)
tensor_dataset_ = TensorDataset(x_st, y_st)
tensor_dataloader_ = DataLoader(tensor_dataset_, batch_size = 5, shuffle = True, num_workers = 0)

def gan(D, S, D_solver, S_solver):
    iter_count = 0
    show_every = 250
    for epoch in range(300):
        for x, y in tensor_dataloader: 
            x = Variable(x).type(dtype)
            y_true = Variable(y).type(dtype)
            x_high = to_high_res(x).type(dtype)
            for x_st, y_st in tensor_dataloader_:
                x_st = Variable(x_st).type(dtype)
                y_st = Variable(y_st).type(dtype)
                x_st_high = to_high_res(x_st).type(dtype)
                
                y_predict,  _ = S(x_high, x)
                y_predict = y_predict.type(dtype)
                _, y_st_pre = S(x_st_high, x_st)
                y_st_predict = D(y_st_pre).type(dtype)
            
                S_solver.zero_grad()
                if epoch < 10:
                    S_error = segmenter_loss(y_predict, y_true, y_st_predict, y_st, 0)
                elif epoch < 35:
                    S_error = segmenter_loss(y_predict, y_true, y_st_predict, y_st, 0.05 * (epoch - 9) / (34 - 9))
                else:
                    S_error = segmenter_loss(y_predict, y_true, y_st_predict, y_st, 0.05)
                S_error.backward(retain_graph = True)
                S_solver.step()
                
                D_solver.zero_grad()
                D_error = discriminator_loss(y_st_predict, y_st)
                D_error.backward()
                D_solver.step()

                if (iter_count % show_every == 0):
                    print('Iter: {}, D: {:.4}, S:{:.4}'.format(iter_count, D_error.data[0], S_error.data[0]))  
                iter_count += 1

def just_segmenter(S, S_solver, just_segmenter_loss, show_every = 250, 
              batch_size=1, num_epochs = 300):

    iter_count = 0
    for epoch in range(300):
        for x, y in tensor_dataloader:  
            x = Variable(x).type(dtype)
            y_true = Variable(y).type(dtype)
            x_high = to_high_res(x).type(dtype)
            y_predict,  _ = S(x_high, x)
            y_predict = y_predict.type(dtype)
            S_solver.zero_grad()
            S_error = just_segmenter_loss(y_predict, y_true)
            S_error.backward(retain_graph = True)
            S_solver.step()
            if (iter_count % 250 == 0):
                print('Iter: {}, S:{:.4}'.format(iter_count, S_error.data[0]))
            iter_count += 1

D_1 = discriminator()
S_1 = Segmenter()

D_solver_1 = get_optimizer(D_1)
S_solver_1 = get_optimizer(S_1)

gan(D_1, S_1, D_solver_1, S_solver_1)

S_2 = Segmenter()
S_solver_2 = get_optimizer(S_2)
just_segmenter(S_2, S_solver_2, just_segmenter_loss)

test_s_predict, _ = S_1(test_s_high, test_s)
test_t_predict, _ = S_1(test_t_high, test_t)
#print just_segmenter_loss(test_s_predict, test_s_y)
print just_segmenter_loss(test_t_predict, test_t_y)
#compute_dsc_precision_recall(test_s_predict, test_s_y, 0.6)
compute_dsc_precision_recall(test_t_predict, test_t_y, 0.4)

test_s_predict, _ = S_2(test_s_high, test_s)
test_t_predict, _ = S_2(test_t_high, test_t)
#print just_segmenter_loss(test_s_predict, test_s_y)
print just_segmenter_loss(test_t_predict, test_t_y)
#compute_dsc_precision_recall(test_s_predict, test_s_y, 0.6)
compute_dsc_precision_recall(test_t_predict, test_t_y, 0.4)
