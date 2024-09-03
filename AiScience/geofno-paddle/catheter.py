import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as Initializer


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
        x_ft = paddle.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft_real = paddle.zeros([batchsize, self.out_channels,x.shape[-1]//2 + 1])
        out_ft_img = paddle.zeros([batchsize, self.out_channels,x.shape[-1]//2 + 1])
        out_ft = paddle.complex(out_ft_real,out_ft_img)
        
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        if output_size == None:
            x = paddle.fft.irfft(out_ft, n=x.shape[-1])
        else:
            x = paddle.fft.irfft(out_ft, n=output_size)

        return x


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
        self.fc0 = nn.Linear(input_channel, self.width, weight_attr=nn.initializer.KaimingUniform(), bias_attr=nn.initializer.KaimingUniform())

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv4 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1D(self.width, self.width, 1, weight_attr=nn.initializer.KaimingUniform(), bias_attr=nn.initializer.KaimingUniform())
        self.w1 = nn.Conv1D(self.width, self.width, 1, weight_attr=nn.initializer.KaimingUniform(), bias_attr=nn.initializer.KaimingUniform())
        self.w2 = nn.Conv1D(self.width, self.width, 1, weight_attr=nn.initializer.KaimingUniform(), bias_attr=nn.initializer.KaimingUniform())
        self.w3 = nn.Conv1D(self.width, self.width, 1, weight_attr=nn.initializer.KaimingUniform(), bias_attr=nn.initializer.KaimingUniform())

        self.fc1 = nn.Linear(self.width, 128, weight_attr=nn.initializer.KaimingUniform(), bias_attr=nn.initializer.KaimingUniform())
        self.fc2 = nn.Linear(128, 1, weight_attr=nn.initializer.KaimingUniform(), bias_attr=nn.initializer.KaimingUniform())

    def _FUNCTIONAL_PAD(self, x, pad, mode='constant', value=0.0, data_format='NCL'):
        if len(x.shape) * 2 == len(pad) and mode == "constant":
            pad = paddle.to_tensor(pad, dtype="float32").reshape(
                (-1, 2)).flip([0]).flatten().tolist()
        return F.pad(x, pad, mode, value, data_format)

    def forward(self, x): # x : 20, 2001, 2
        
        x = self.fc0(x) # [20, 2001, 64]
        x = paddle.transpose(x, perm=[0, 2, 1])# [20, 64, 2001]
        # pad the domain if input is non-periodic
        x = self._FUNCTIONAL_PAD(x, [0, self.padding])
        
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


def catheter_mesh_1d_total_length(L_x, L_p, x2, x3, h, N_s):
    x1 = -0.5*L_p
    # ncy = 20
    
    n_periods = torch.floor(L_x / L_p)
    L_x_last_period = L_x - n_periods*L_p
    L_p_s = ((x1 + L_p) + (0 - x3) + torch.sqrt((x2 - x1)**2 + h**2) + torch.sqrt((x3 - x2)**2 + h**2))
    L_s = L_p_s*n_periods + Lx2length(L_x_last_period, L_p, x1, x2, x3, h)
    
    # from 0
    d_arr = torch.linspace(0, 1, N_s) * L_s
    
    # TODO do not compute gradient for floor
    period_arr = torch.floor(d_arr / L_p_s).detach()
    d_arr -= period_arr * L_p_s

    
    xx, yy = d2xy(d_arr, L_p, x1, x2, x3, h)
        
    xx = xx - period_arr*L_p
    
    
    X_Y = torch.zeros((1, N_s, 2), dtype=torch.float).to(device)
    X_Y[0, :, 0], X_Y[0, :, 1] = xx, yy
    return X_Y, xx, yy

def Lx2length(L_x, L_p, x1, x2, x3, h):
    l0, l1, l2, l3 = -x3, torch.sqrt((x2-x3)**2 + h**2), torch.sqrt((x1-x2)**2 + h**2), L_p+x1
    if L_x < -x3:
        l = L_x
    elif L_x < -x2:
        l = l0 + l1*(L_x + x3)/(x3-x2)
    elif L_x < -x1:
        l = l0 + l1 + l2*(L_x + x2)/(x2-x1)
    else:
        l = l0 + l1 + l2 + L_x+x1

    return l

def d2xy(d, L_p, x1, x2, x3, h):
    
    p0, p1, p2, p3 = torch.tensor([0.0,0.0]), torch.tensor([x3,0.0]), torch.tensor([x2, h]), torch.tensor([x1,0.0])
    v0, v1, v2, v3 = torch.tensor([x3-0,0.0]), torch.tensor([x2-x3,h]), torch.tensor([x1-x2,-h]), torch.tensor([-L_p-x1,0.0])
    l0, l1, l2, l3 = -x3, torch.sqrt((x2-x3)**2 + h**2), torch.sqrt((x1-x2)**2 + h**2), L_p+x1
    
    xx, yy = torch.zeros(d.shape), torch.zeros(d.shape)
    ind = (d < l0)
    xx[ind] = d[ind]*v0[0]/l0 + p0[0]
    yy[ind] = d[ind]*v0[1]/l0 + p0[1]
    
    ind = torch.logical_and(d < l0 + l1, d>=l0)
    xx[ind] = (d[ind]-l0)*v1[0]/l1 + p1[0] 
    yy[ind] = (d[ind]-l0)*v1[1]/l1 + p1[1]
    
    ind = torch.logical_and(d < l0 + l1 + l2, d>=l0 + l1)
    xx[ind] = (d[ind]-l0-l1)*v2[0]/l2 + p2[0]
    yy[ind] = (d[ind]-l0-l1)*v2[1]/l2 + p2[1]
    
    ind = (d>=l0 + l1 + l2)
    xx[ind] = (d[ind]-l0-l1-l2)*v3[0]/l3 + p3[0]
    yy[ind] = (d[ind]-l0-l1-l2)*v3[1]/l3 + p3[1]
    

    return xx, yy