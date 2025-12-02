import torch.nn as nn
import torch.nn.functional as F

class Complex3DCNN(nn.Module):
    def __init__(self, input_shape=(128,128,128)):
        super().__init__()
        D,H,W = input_shape
        
        '''
            nn.Conv3d(in_channels, out_channels, kernel_size padding)
            
            In the OASIS dataset, MRI images have 1 channel: T1, i.e.
            relaxation time of tissues after a radiofrequency pulse. 
            
            Padding formula:            
            out_size = (in_size+2*padding-kernel_size)/stride+1
            If stride=1 and we want out_size=in_size=128:
            2*padding = kernel_size-1 -> padding = (kernel_size-1)/2
            
            By default, padding values are 0. 
        '''
                
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        
        '''
            MaxPool3d(kernel_size, stride=None, padding=0)
            If stride=None, stride_size will be the same as kernel_size.
            Choosing kernel_size=2 thus slices input dimension in half.
        '''
        
        self.pool  = nn.MaxPool3d(2)

        '''
            Fully connected layer. 
            The input has 64 channels from the last convolution, 
            with spatial dimensions reduced by two maxpoolings (divided by 4). 
            The output has 128 neurons.
        '''
        
        self.fc1 = nn.Linear(64*(D//4)*(H//4)*(W//4), 128)

        '''
            Dropout(p)
            During training, zeros some of the elements of the input tensor 
            with probability p.
        '''
        self.fc2 = nn.Linear(128, 2)  # output: 2 classes 

    def forward(self, x):        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        '''
            fc needs a 2d input: (batch_size, num_features).
            After cnn, x has shape (batch_size, channels, D, H, W).
            Thus we must flatten x.
        '''
        
        x = x.view(x.size(0), -1) # x is a torch tensor: x.size(0) is always batch_size 
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Simple3DCNN(nn.Module):
    def __init__(self, input_shape=(128,128,128)):
        super().__init__()
        D,H,W = input_shape
                
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool3d(2)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(32, 128)

        '''
            Dropout(p)
            During training, zeros some of the elements of the input tensor 
            with probability p.
        '''
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 2)  # output: 2 classes 

    def forward(self, x):       
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)