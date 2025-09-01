'''使用LSTM'''

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class GestureLSTM(nn.Module):
    '''基础LSTM模型'''
    def __init__(self, input_dim = 96, hidden_dim = 256, num_layers=3, num_classes= 10 , dropout = 0.2  ):
        super(GestureLSTM,self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size= hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,num_classes)
        )
    
    def forward(self,x):
        
        lstm_output,_ = self.lstm(x)
        
        last_output = lstm_output[:,-1,:]
        
        output = self.classifier(last_output)
        
        return output
    
    