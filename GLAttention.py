import torch
import torch.nn as nn
import torch.nn.functional as F

class Horizontal_Attention(nn.Module):
    def __init__(self,in_dim, ratio):
        super(Horizontal_Attention,self).__init__()
        inter_dim = in_dim // ratio
        self.in_channel = in_dim
        self.q_conv = nn.Conv2d(in_channels=in_dim,out_channels=inter_dim,kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels=in_dim,out_channels=inter_dim,kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels=in_dim,out_channels=inter_dim,kernel_size=1)
        self.delta = nn.Parameter(torch.Tensor([0.2]))

    def forward(self,input):

        B,C,H,W = input.size()

        query = self.q_conv(input)
        CQ = query.size(1)
        query = query.permute(0, 1, 3, 2).view(B, CQ*W, -1)#[B,W*C',H]
        # query = query.permute(0, 2, 1, 3).view(B, -1,CQ*W)
        # query = query.permute(0, 2, 1)#[B, W*C', H]

        key = self.k_conv(input)
        key = key.permute(0, 2, 1, 3).view(B, -1, CQ*W)#[B, H, W*C']

        value = self.v_conv(input)#[B, C', H, W]
        value = value.permute(0, 2, 1, 3).view(B, -1, CQ*W)#[B, H, W*C']


        Pro = F.softmax(torch.bmm(query,key),dim=1) #[B, W*C', W*C']

        Attention_map = torch.bmm(value,Pro).permute(0,2,1)
        Weight = Attention_map.view(B,CQ,H,W)

        out = self.q_conv(input)
        out = Weight*self.delta + out

        return  out

class Vertical_Attention(nn.Module):
    def __init__(self,in_dim):
        super(Vertical_Attention, self).__init__()
        self.in_channel = in_dim
        self.q_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.delta = nn.Parameter(torch.Tensor([0.2]))
    def forward(self,input):
        B,C,H,W = input.size()
        query = self.q_conv(input)
        CQ = query.size(1)
        query = query.view(B, CQ*H, W)#[B, C'*H, W]

        key = self.k_conv(input).view(B, CQ*H, W).permute(0, 2, 1)#[B, W, C'*H]

        value = self.v_conv(input).view(B, CQ*H, W).permute(0, 2, 1)#[B, W, C'*H]

        pro = F.softmax(torch.bmm(query,key),dim=1)#[B, C'*H, C'*H]

        AttentionMap = torch.bmm(value, pro).permute(0, 2, 1).view(B, CQ, H, W)

        out = self.k_conv(input)
        out = self.delta*AttentionMap + out
        return  out


class AttentionBlock(nn.Module):
    def __init__(self,ic):
        super(AttentionBlock,self).__init__()
        self.HA = Horizontal_Attention(ic)
        self.VA = Vertical_Attention(ic)
        self.delta = nn.Parameter(torch.Tensor([0.1]))


    def forward(self,x):
        HA = self.HA(x)
        DA = self.VA(x)
        weight = HA + DA

        out = x + self.delta*weight
        return  out
