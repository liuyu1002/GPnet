import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SITM(nn.Module):
    def __init__(self,in_channel,hidden_channel):
        super(SITM,self).__init__()
        self.reset = nn.Conv2d(in_channels=in_channel*2,out_channels=in_channel,kernel_size=1)
        self.in_channel = in_channel
        self.conv = nn.Conv2d(in_channels=hidden_channel,out_channels=in_channel,kernel_size=3,stride=2,padding=1)
        self.delta = nn.Parameter(torch.Tensor([0.1]))


    def forward(self,input,pre_state):
        #  get batch and spatial sizes
        batch = input.size(0)
        spatial = input.size(3)
        #  generata empty perstate, if Note is provided
        if pre_state is None:
            state_size = [batch, self.in_channel,spatial,spatial]
            if torch.cuda.is_available():
                pre_state = Variable(torch.zeros(state_size)).cuda()
            else:
                pre_state = Variable(torch.zeros(state_size))
        else:
            pre_state = self.conv(pre_state)

        pre_state = torch.max_pool2d(pre_state,kernel_size=1)
        pre_state = nn.MaxPool2d(1)(pre_state)
        if pre_state.size()[3] != input.size()[3]:

            diffY = pre_state.size()[2] - input.size()[2]
            diffX = pre_state.size()[3] - input.size()[3]

            input = nn.functional.pad(input, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        stacked_inputs = torch.cat([input, pre_state], dim=1)
        out = self.reset(stacked_inputs)
        reset  = torch.sigmoid(out)

        update = torch.sigmoid(input + reset * pre_state)
        new_state = input+ self.delta*(update * input)

        return new_state

class MAM(nn.Module):# channel
    def __init__(self,):
        super(MAM,self).__init__()
        self.delta = nn.Parameter(torch.Tensor([0.1]))

    def forward(self,input,state):
        if state.size()[3] != input.size()[3]:

            diffY = state.size()[2] - input.size()[2]
            diffX = state.size()[3] - input.size()[3]

            input = nn.functional.pad(input, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        resnet_gata = torch.sigmoid(input + state)
        resnet = resnet_gata*state
        weight = input * resnet_gata
        weigt = weight + resnet
        out = input * self.delta*weigt

        return out


class Globel_Attention(nn.Module):
    def __init__(self,c,s,k):
        super(Globel_Attention,self).__init__()
        # self.k = 64
        # self.c = x.size(1)
        # self.s = x.size(2)*x.size(3)
        self.linear_c = nn.Linear(c,k)
        self.linear_s = nn.Linear(s,k)
        self.delta = nn.Parameter(torch.Tensor([0.1]))

    def forward(self,input):
        # c = input.size(1)
        # s = input.size(2) * input.size(3)
        # linear_c = nn.Linear(c, k).cuda()
        # linear_s = nn.Linear(s, k).cuda()

        B,C,H,W = input.size()
        x = input.view(B,C,-1).contiguous() #[B,C,H*W]
        # print(x.shape)
        Attention_s = self.linear_s(x) #[B,C,k]
        x = x.permute(0,2,1).contiguous() #[B,H*W,C]
        Attention_c = self.linear_c(x).permute(0,2,1).contiguous() #[B,k,H*W]
        out = torch.bmm(Attention_s,Attention_c)#[B,C,H*W]
        out = out.view(B,C,H,W).contiguous()
        out = input + self.delta*out


        return  out

class Convblock(nn.Module):
    def __init__(self,ic,oc,ks):
        # ic: input channels
        # oc: ouput channels
        # ks: kernel size
        super(Convblock,self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(ic,oc,kernel_size=ks,padding=(ks-1)//2),
            nn.BatchNorm2d(oc),
            nn.LeakyReLU(),
            nn.Conv2d(oc,oc,kernel_size=ks,padding=(ks-1)//2),
            nn.BatchNorm2d(oc),
            nn.LeakyReLU()
        )


    def forward(self,x):
        y = self.left(x)
        return  y

class decoderunit(nn.Module):
    def __init__(self,ic1,ic2,ot,ks):
        # ic: input channels
        # oc: ouput channels
        # ks: kernel size
        super(decoderunit,self).__init__()
        self.left=nn.Sequential(
            nn.ConvTranspose2d(ic1,ot//2,ks,padding=1),
            nn.BatchNorm2d(ot//2),
            nn.ReLU(),
            nn.ConvTranspose2d(ot//2,ot,kernel_size=2,stride=2),
            nn.BatchNorm2d(ot),
            nn.ReLU()
        )
        self.conv1 = nn.Conv2d(ot,ot//2,kernel_size=1)
        self.conv2 = nn.Conv2d(ic2,ot//2,kernel_size=1)
    def forward(self,x1,x2):

        x1 = self.left(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2))

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x = torch.cat([x1,x2],dim=1)
        out = F.relu(x)

        return out










