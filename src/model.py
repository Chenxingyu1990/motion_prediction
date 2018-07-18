import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from common import action_utils
from core import op
import ipdb


class CNNRegressPose(nn.Module):
    def __init__(self, n_in, n_out):
        super(CNNRegressPose, self).__init__()
        self.conv1 = nn.Conv1d(54, 512, kernel_size = 25, padding = 12)   # 50, 100 same    
        self.relu1 = nn.LeakyReLU(0.2)
        self.pooling1 = nn. MaxPool1d(2) #
        self.dropout = nn.Dropout(0.1)
        self.deconv2 = nn.ConvTranspose1d(512, 54, kernel_size = 25 , stride = 2, padding = 12, output_padding = 1 )
        self.relu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv1d(50, 512, kernel_size = 25, padding = 12 )
        self.relu3 = nn.LeakyReLU(0.2)
        self.pooling2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.1)
        self.deconv4 = nn.ConvTranspose1d(512, 50, kernel_size = 25, stride = 2, padding = 12, output_padding = 1)
        self.relu4 = nn.LeakyReLU(0.2)
        self.linear5 = nn.Linear(54,54)
    def forward(self, input):

        x0 = input.permute(0, 2, 1)   
        x1 = self.conv1(x0)
        x1 = self.relu1(x1)
        x2 = self.pooling1(x1)
        x2 = self.dropout(x2)
        x3 = self.deconv2(x2)
        x3 = self.relu2(x3)    
        x3 = x3.permute(0,2,1)

        x4 = self.conv3(x3)
        x4 = self.relu3(x4)
        x5 = self.pooling2(x4)
        x5 = self.dropout2(x5)
        x6 = self.deconv4(x5)
        x6 = self.relu4(x6)
        x7 = x6 + x3
        x8 = self.linear5(x7)
        return x8
'''
class LSTMRegressPose(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, input_len, future):
        super(LSTMRegressPose, self).__init__()
        self.n_hidden = n_hidden
        self.input_len = input_len
        self.future = future
        self.rnn = nn.LSTMCell(n_in,n_hidden)
        self.linear = nn.Linear(n_hidden, n_out)

        
    def forward(self, input):
        outputs = []  
      
        h_t = torch.zeros(input.size(0), self.n_hidden, dtype=torch.float).cuda()
        c_t = torch.zeros(input.size(0), self.n_hidden, dtype=torch.float).cuda()
        for i, input_t in enumerate(input.chunk(self.input_len, dim=1)):
            input_t = input_t.view(input_t.shape[0], input_t.shape[2])
            h_t, c_t = self.rnn(input_t, (h_t, c_t))
          
            output = self.linear(h_t)
            output = output + input_t
            #outputs += [output]

        for i in range(self.future):#  predict the future
   
            input_tt = output
            h_t, c_t = self.rnn(input_tt, (h_t, c_t))
            output = self.linear(h_t)            
            output = output + input_tt
            outputs += [output]
            
        outputs = torch.stack(outputs, 1).squeeze(2)

        return outputs
'''

class LSTMRegressPose(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, input_len, future):
        super(LSTMRegressPose, self).__init__()
        self.n_hidden = n_hidden
        self.input_len = input_len
        self.future = future
        self.rnn1 = nn.GRUCell(n_in,n_hidden)
        self.linear = nn.Linear(n_hidden, n_out)

        
    def forward(self, input):
        outputs = []  
      
        h_t = torch.zeros(input.size(0), self.n_hidden, dtype=torch.float).cuda()
        for i, input_t in enumerate(input.chunk(self.input_len, dim=1)):
            input_t = input_t.view(input_t.shape[0], input_t.shape[2])
            h_t= self.rnn1(input_t, h_t)
        
         
    
        input_tt = input_t
        input_tt = torch.autograd.Variable(input_tt, requires_grad = True)
        output = input_t
        for i in range(self.future):#  predict the future
   
            h_t = self.rnn1(output, h_t)
            output1 = self.linear(h_t)            
            output =  input_tt
            
            
            outputs += [output]
            
        outputs = torch.stack(outputs, 1).squeeze(2)

        return outputs
        
        
        
        
        
        
        
        
        
        
        
        
        