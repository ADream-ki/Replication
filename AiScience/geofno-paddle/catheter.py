import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as Initializer

import ppsci
from ppsci.arch import base


################################################################
# fourier layer OK
################################################################
class SpectralConv1d(nn.Layer):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        self.scale = (1 / (in_channels*out_channels))

        self.weights1_real = self.create_parameter([in_channels, out_channels, self.modes1], 
                                                   attr=Initializer.Assign(self.scale*paddle.rand(shape=[in_channels, out_channels, modes1])))
        self.weights1_img = self.create_parameter([in_channels, out_channels, self.modes1], 
                                                  attr=Initializer.Assign(self.scale*paddle.rand(shape=[in_channels, out_channels, modes1])))
        self.weights1 = paddle.complex(self.weights1_real,self.weights1_img)
        # tmp = paddle.ParamAttr(initializer=Initializer.Normal(0.0j, self.scale))
        # self.weights1 = paddle.create_parameter(shape=(self.in_channels, self.out_channels, self.modes1), dtype=paddle.complex64,attr=tmp)
        # print(self.weights1)

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return paddle.einsum("bix,iox->box", input, weights)

    def forward(self, x, output_size=None):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfftn(x, axes=[2])

        # Multiply relevant Fourier modes
        out_ft_real = paddle.zeros([batchsize, self.out_channels,x.shape[-1]//2 + 1])
        out_ft_img = paddle.zeros([batchsize, self.out_channels,x.shape[-1]//2 + 1])
        out_ft = paddle.complex(out_ft_real,out_ft_img)
        # out_ft = paddle.zeros(batchsize, self.in_channels, x.shape[-1]//2 + 1, dtype=paddle.complex64)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        if output_size == None:
            x = paddle.fft.irfft(out_ft, n=x.shape[-1])
        else:
            x = paddle.fft.irfft(out_ft, n=output_size)

        return x


# OK 
class FNO1d(nn.Layer):
    def __init__(self, modes, width, padding=100, input_channel=2,
                 output_np=2001):
        super(FNO1d, self).__init__()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.output_np = output_np
        self.modes1 = modes
        self.width = width
        self.padding = padding
        self.fc0 = nn.Linear(input_channel, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv4 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1D(self.width, self.width, 1)
        self.w1 = nn.Conv1D(self.width, self.width, 1)
        self.w2 = nn.Conv1D(self.width, self.width, 1)
        self.w3 = nn.Conv1D(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def _FUNCTIONAL_PAD(self, x, pad, mode='constant', value=0.0, data_format='NCL'):
        if len(x.shape) * 2 == len(pad) and mode == "constant":
            pad = paddle.to_tensor(pad, dtype="int32").reshape(
                (-1, 2)).flip([0]).flatten().tolist()
        return F.pad(x, pad, mode, value, data_format)

    def forward(self, x): # x : 20, 2001, 2
        
        x = self.fc0(x) # [20, 2001, 64]
        x = paddle.transpose(x, perm=[0, 2, 1])# [20, 64, 2001]
        # pad the domain if input is non-periodic
        # x = self._FUNCTIONAL_PAD(x, [0, self.padding])
        # [20, 64, 2101]
        x = F.pad(x, pad= [0, self.padding], mode='constant', value=0.0, data_format='NCL')

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x=x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x)

        x = x[..., :-self.padding]
        # x(batch, channel, y)
        # print(x.shape) [20, 64, 2001]
        x1 = self.conv4(x, self.output_np)
        # print(x.shape) [20, 64, 2001]
        # print(x1.shape) [20, 64, 2001]
        x2 = F.interpolate(x, size=[self.output_np],
                           mode='linear', align_corners=True)
        x = x1 + x2
        # x(batch, channel, 2001)
        x = x.transpose(perm=[0, 2, 1])
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        # x = x.transpose(perm=[0, 2, 1])
        # x = self.fc1(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = paddle.to_tensor(np.linspace(0, 1, size_x), dtype='float')
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)



# analysis
class Net(nn.Layer):
    def __init__(self, n):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, n)
        self.fc2 = nn.Linear(n, n)
        self.fc3 = nn.Linear(n, n)
        self.fc4 = nn.Linear(n, 1)

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = F.gelu(x)
        x = self.fc4(x)
        return x
    

