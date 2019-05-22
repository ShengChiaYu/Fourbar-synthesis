import os
import sys
from math import pi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataset import FDs

class PosLoss(nn.Module):
    def __init__(self, pos_num):
        super(PosLoss,self).__init__()
        self.pos_num = pos_num

    def compute_pos(self, pred, pos, th2_min, th2_max):
        step = (th2_max - th2_min) / (self.pos_num - 1)
        for i in range(self.pos_num):
            th2 = th2_min + step * i
            k1 = pred[:,0]**2 + pred[:,1]**2 + pred[:,2]**2 - pred[:,3]**2 - 2*pred[:,0]*pred[:,1]*torch.cos(th2-pred[:,4])
            k2 = 2*pred[:,0]*pred[:,2]*torch.cos(pred[:,4]) - 2*pred[:,1]*pred[:,2]*torch.cos(th2)
            k3 = 2*pred[:,0]*pred[:,2]*torch.sin(pred[:,4]) - 2*pred[:,1]*pred[:,2]*torch.sin(th2)
            a = k1 + k2
            b = -2 * k3
            c = k1 -k2

            D = torch.sqrt(b**2 - 4*a*c)
            nan_mask = torch.isnan(D)
            x_1 = (-b + D) / (2 * a) # x_1 and x_2 = tan((1/2)*th3)

            th3_1 = 2*torch.atan(x_1)

            p1x = pred[:,1]*torch.cos(th2) + pred[:,5]*torch.cos(pred[:,6]+th3_1) + pred[:,7]
            p1y = pred[:,1]*torch.sin(th2) + pred[:,5]*torch.sin(pred[:,6]+th3_1) + pred[:,8]
            p1x[nan_mask] = 0
            p1y[nan_mask] = 0
            pos[:,2*i] = p1x
            pos[:,2*i+1] = p1y
        return pos

    def path_gen_open(self, L, th1, r, alpha, n, x0, y0, valid_target):
        '''
        L: (tensor) size(batchsize,4) [r1,r2,r3,r4]
        th1: (tensor) size(batchsize,1) [th1]
        r: (tensor) size(batchsize,1) [r5]
        alpha: (tensor) size(batchize,1) [alpha]
        n: (int) [60/360]
        x0: (tensor) size(batchize,1) [x0]
        y0: (tensor) size(batchize,1) [y0]
        valid_target: (tensor) size(batchize,2*n) [[x1,y1,x2,y2,...],[],...]
        '''
        # combine the linkages
        bs = L.size(0)
        valid_pred = torch.cuda.FloatTensor(bs,9) # [bs,9]
        valid_pred[:,:4] = L
        valid_pred[:,4] = th1
        valid_pred[:,5] = r
        valid_pred[:,6] = alpha
        valid_pred[:,7] = x0
        valid_pred[:,8] = y0

        # Limit of rotation angle of input linkage
        cond_1_mask_b = (L[:,0] + L[:,1]) - (L[:,2] + L[:,3]) > 0
        cond_1_mask_s = (L[:,0] + L[:,1]) - (L[:,2] + L[:,3]) <= 0
        cond_2_mask_b = abs(L[:,0] - L[:,1]) - abs(L[:,2] - L[:,3]) >= 0
        cond_2_mask_s = abs(L[:,0] - L[:,1]) - abs(L[:,2] - L[:,3]) < 0

        # Upper limit exists
        up_mask = cond_1_mask_b * cond_2_mask_b
        up_mask_pred = up_mask.unsqueeze(-1).expand_as(valid_pred)
        up_mask_target = up_mask.unsqueeze(-1).expand_as(valid_target)
        up_pred = valid_pred[up_mask_pred].view(-1,9)
        up_target = valid_target[up_mask_target].view(-1,2*n)
        up_th2_max = torch.acos((up_pred[:,0]**2 + up_pred[:,1]**2 - (up_pred[:,2] + up_pred[:,3])**2) /
                                (2*up_pred[:,0]*up_pred[:,1]))

        # Lower limit exists
        low_mask = cond_1_mask_s * cond_2_mask_s
        low_mask_pred = low_mask.unsqueeze(-1).expand_as(valid_pred)
        low_mask_target = low_mask.unsqueeze(-1).expand_as(valid_target)
        low_pred = valid_pred[low_mask_pred].view(-1,9)
        low_target = valid_target[low_mask_target].view(-1,2*n)
        low_th2_min = torch.acos((low_pred[:,0]**2 + low_pred[:,1]**2 - (low_pred[:,2] - low_pred[:,3])**2) /
                                (2*low_pred[:,0]*low_pred[:,1]))

        # Both limit exist
        bo_mask = cond_1_mask_b * cond_2_mask_s
        bo_mask_pred = bo_mask.unsqueeze(-1).expand_as(valid_pred)
        bo_mask_target = bo_mask.unsqueeze(-1).expand_as(valid_target)
        bo_pred = valid_pred[bo_mask_pred].view(-1,9)
        bo_target = valid_target[bo_mask_target].view(-1,2*n)
        bo_th2_max = torch.acos((bo_pred[:,0]**2 + bo_pred[:,1]**2 - (bo_pred[:,2] + bo_pred[:,3])**2) /
                                (2*bo_pred[:,0]*bo_pred[:,1]))
        bo_th2_min = torch.acos((bo_pred[:,0]**2 + bo_pred[:,1]**2 - (bo_pred[:,2] - bo_pred[:,3])**2) /
                                (2*bo_pred[:,0]*bo_pred[:,1]))

        # No limit exists
        no_mask = cond_1_mask_s * cond_2_mask_b
        no_mask_pred = no_mask.unsqueeze(-1).expand_as(valid_pred)
        no_mask_target = no_mask.unsqueeze(-1).expand_as(valid_target)
        no_pred = valid_pred[no_mask_pred].view(-1,9)
        no_target = valid_target[no_mask_target].view(-1,2*n)
        no_th2_min = torch.cuda.FloatTensor(no_pred.size(0))
        no_th2_min.zero_()
        no_th2_max = torch.cuda.FloatTensor(no_pred.size(0))
        no_th2_max[:] = 2 * pi

        # Calculate the positions of coupler curve and losses by different input angles
        up_pos = torch.cuda.FloatTensor(up_target.size())
        up_pos.zero_()
        if up_target.size(0):
            up_pos = self.compute_pos(up_pred, up_pos, -up_th2_max, up_th2_max)
        up_loss = F.mse_loss(up_pos, up_target, reduction='sum')

        low_pos = torch.cuda.FloatTensor(low_target.size())
        low_pos.zero_()
        if low_target.size(0):
            low_pos = self.compute_pos(low_pred, low_pos, low_th2_min, 2*pi-low_th2_min)
        low_loss = F.mse_loss(low_pos, low_target, reduction='sum')

        bo_pos = torch.cuda.FloatTensor(bo_target.size())
        bo_pos.zero_()
        if bo_target.size(0):
            bo_pos = self.compute_pos(bo_pred, bo_pos, bo_th2_min, bo_th2_max)
        bo_loss = F.mse_loss(bo_pos, bo_target, reduction='sum')

        no_pos = torch.cuda.FloatTensor(no_target.size())
        no_pos.zero_()
        if no_target.size(0):
            no_pos = self.compute_pos(no_pred, no_pos, no_th2_min, no_th2_max)
        no_loss = F.mse_loss(no_pos, no_target, reduction='sum')

        return up_loss, low_loss, bo_loss, no_loss # (up_loss + low_loss + bo_loss + no_loss)

    def forward(self,pred_tensor,target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,5) [r1,r3,r4,r5,thc]
        target_tensor: (tensor) size(batchsize,60/360)
        '''
        N = pred_tensor.size()[0]
        # put r2=1 into pred_params
        pred_params = torch.cuda.FloatTensor(N, pred_tensor.size(1)+1)
        pred_params[:,0] = pred_tensor[:,0]    # r1
        pred_params[:,1] = 1.                  # r2
        pred_params[:,2:] = pred_tensor[:,1:]  # r3 r4 r5 thc

        # find valid and invalid masks of linkages
        max_link = torch.max(pred_params[:,:4],dim=1)[0]
        valid_mask = 2*max_link < torch.sum(pred_params[:,:4], dim=1)
        invalid_mask = 2*max_link >= torch.sum(pred_params[:,:4], dim=1)

        # linkages
        valid_mask_params = valid_mask.unsqueeze(-1).expand_as(pred_params)
        invalid_mask_params = invalid_mask.unsqueeze(-1).expand_as(pred_params)
        valid_pred = pred_params[valid_mask_params].view(-1,6) # [, 6]
        invalid_pred = pred_params[invalid_mask_params].view(-1,6) # [, 6]

        # postions
        valid_mask_pos = valid_mask.unsqueeze(-1).expand_as(target_tensor)
        invalid_mask_pos = invalid_mask.unsqueeze(-1).expand_as(target_tensor)
        valid_target = target_tensor[valid_mask_pos].view(-1,2*self.pos_num) # [, 120/720]
        invalid_target = target_tensor[invalid_mask_pos].view(-1,2*self.pos_num) # [, 120/720]

        # loss of invalid linkages
        invalid_pos = torch.cuda.FloatTensor(invalid_target.size())
        invalid_pos.zero_()
        invalid_loss = F.mse_loss(invalid_pos, invalid_target, reduction='sum')

        # compute positions of valid linkages
        if valid_pred.size(0):
            th1 = torch.cuda.FloatTensor(valid_pred.size(0))
            th1.zero_()
            x0 = torch.cuda.FloatTensor(valid_pred.size(0))
            x0.zero_()
            y0 = torch.cuda.FloatTensor(valid_pred.size(0))
            y0.zero_()
            up_loss, low_loss, bo_loss, no_loss = self.path_gen_open(valid_pred[:,:4], th1, valid_pred[:,4], valid_pred[:,5],
                                                                     self.pos_num, x0, y0, valid_target)
        else:
            up_pos = torch.cuda.FloatTensor(valid_target.size())
            up_pos.zero_()
            up_loss = F.mse_loss(up_pos, valid_target, reduction='sum')
            low_pos = torch.cuda.FloatTensor(valid_target.size())
            low_pos.zero_()
            low_loss = F.mse_loss(low_pos, valid_target, reduction='sum')
            bo_pos = torch.cuda.FloatTensor(valid_target.size())
            bo_pos.zero_()
            bo_loss = F.mse_loss(bo_pos, valid_target, reduction='sum')
            no_pos = torch.cuda.FloatTensor(valid_target.size())
            no_pos.zero_()
            no_loss = F.mse_loss(no_pos, valid_target, reduction='sum')
        invalid_loss.requires_grad = True

        # valid_loss = up_loss + low_loss + bo_loss + no_loss
        return invalid_loss/N, up_loss/N, low_loss/N, bo_loss/N, no_loss/N # (invalid_loss + valid_loss) / N


def test():
    root = os.path.join('data','60positions')
    y_file = '_pos.csv'
    print('root:', root, ', y file:', y_file)

    testset = FDs(root=root, x_file='y_test_param.csv', y_file='y_test'+y_file, transform=False)
    testset_loader = DataLoader(testset, batch_size=5, shuffle=False, num_workers=6)

    dataiter = iter(testset_loader)
    outputs, targets = dataiter.next()

    criterion = PosLoss(60)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    outputs = outputs.to(device)
    targets = targets.to(device)
    invalid_loss, up_loss, low_loss, bo_loss, no_loss = criterion(outputs, targets)
    print(invalid_loss, up_loss, low_loss, bo_loss, no_loss)


if __name__ == '__main__':
    test()
