import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlockNT_1d(nn.Module):
    '''
    Convolutional block with NTK initialization
    '''

    def __init__(self, input_ch, h, filter_size, stride):

        super().__init__()
        
        self.w = nn.Parameter(
            torch.randn(h, input_ch, filter_size)
        )
        self.b = nn.Parameter(torch.randn(h))
        self.stride = stride
        self.f = filter_size
        
        
        tmp = (filter_size)%2
        
        if tmp !=0:
            self.pool_size_left  = ((self.w.size(-1) - 1) // 2)
            self.pool_size_right  = ((self.w.size(-1) - 1) // 2)
        else:
            self.pool_size_left = ((self.w.size(-1)) // 2) -1
            self.pool_size_right = ((self.w.size(-1)) // 2)  
        
    def forward(self, x):

             
        y = F.pad(x, (self.pool_size_left, self.pool_size_right), mode='circular')
        
        
        h = self.w[0].numel()
        y = F.conv1d(y, self.w / h ** .5,
                     bias=self.b,      
                     stride=self.stride)
        
        y = (self.f**.5)*F.relu(y)
        
        return y





class ConvNetGAPMF_1d(nn.Module):
    '''
    Convolutional neural network with MF initialization and global average
    pooling
    '''

    def __init__(
            self, alpha_tilde, n_blocks, input_ch, h, idx_strides,
            filter_size, stride_int, out_dim):

        super().__init__()
        
        #stacking up layes with stride, after the first layer
        bb = 0
        list_conv = []
        for aa in range(1,n_blocks ): 
            if aa == idx_strides[bb]:
                
                list_conv.append(ConvBlockNT_1d(h ,h, filter_size, stride_int))
                if bb< len(idx_strides)-1:
                    bb+=1
            else:

                list_conv.append(ConvBlockNT_1d(h ,h, filter_size, 1))

        
        self.conv = nn.Sequential( ConvBlockNT_1d(input_ch, h, filter_size, stride_int),
                                  *list_conv
                                  )
  
        self.beta = nn.Parameter(torch.randn(h, out_dim))
        self.alpha_tilde = alpha_tilde
        self.biasFinal = nn.Parameter(torch.randn(1))
    def forward(self, x):

        y = self.conv(x)
        y = y.mean(dim =-1)
        y = ( self.biasFinal + (y @ self.beta)/ self.beta.size(0))* self.alpha_tilde
        
        return y