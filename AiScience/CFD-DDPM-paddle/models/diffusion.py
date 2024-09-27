import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = paddle.exp(paddle.arange(half_dim, dtype='float32') * - emb)
    emb = timesteps.astype('float32').unsqueeze(1) * emb.unsqueeze(0)
    emb = paddle.concat([paddle.sin(emb), paddle.cos(emb)], 1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = paddle.nn.functional.pad(emb, [0, 1, 0, 0])
    return emb


def spatial_fold(input, fold):
    if fold == 1:
        return input

    batch, channel, height, width = input.shape
    h_fold = height // fold
    w_fold = width // fold

    return (
        input.reshape((batch, channel, h_fold, fold, w_fold, fold))
        .transpose((0, 1, 3, 5, 2, 4))
        .reshape((batch, -1, h_fold, w_fold))
    )


def spatial_unfold(input, unfold):
    if unfold == 1:
        return input

    batch, channel, height, width = input.shape
    h_unfold = height * unfold
    w_unfold = width * unfold

    return (
        input.reshape((batch, -1, unfold, unfold, height, width))
        .transpose((0, 1, 4, 2, 5, 3))
        .reshape((batch, -1, h_unfold, w_unfold))
    )


def nonlinearity(x):
    # swish
    return x*paddle.nn.functional.sigmoid(x)


def Normalize(in_channels):
    return paddle.nn.GroupNorm(num_groups=32, num_channels=in_channels, epsilon=1e-6)


class Upsample(nn.Layer):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = paddle.nn.Conv2D(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = paddle.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Layer):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in paddle conv, must do it ourselves
            self.conv = paddle.nn.Conv2D(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = [0, 1, 0, 1]
            x = paddle.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = paddle.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Layer):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, use_scale_shift_norm=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = paddle.nn.Conv2D(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = paddle.nn.Linear(temb_channels,
                                         out_channels * 2 if use_scale_shift_norm else out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = paddle.nn.Dropout(dropout)
        self.conv2 = paddle.nn.Conv2D(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = paddle.nn.Conv2D(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = paddle.nn.Conv2D(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)
        
        self.use_scale_shift_norm = use_scale_shift_norm

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        emb = self.temb_proj(nonlinearity(temb)).unsqueeze(-1).unsqueeze(-1)
        if self.use_scale_shift_norm:#false
            shift, scale = emb.split(2, 1)
            h = self.norm2(h) * (1 + scale) + shift
        else:
            h = h + emb
            h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = paddle.nn.Conv2D(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = paddle.nn.Conv2D(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = paddle.nn.Conv2D(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = paddle.nn.Conv2D(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape([b, c, h*w])
        q = q.transpose([0, 2, 1])   # b,hw,c
        k = k.reshape([b, c, h*w])  # b,c,hw
        w_ = paddle.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = paddle.nn.functional.softmax(w_, 2)

        # attend to values
        v = v.reshape([b, c, h*w])
        w_ = w_.transpose([0, 2, 1])   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = paddle.bmm(v, w_)
        h_ = h_.reshape([b, c, h, w])

        h_ = self.proj_out(h_)

        return x+h_


class Model(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps
        
        use_scale_shift_norm = config.model.use_scale_shift_norm if 'use_scale_shift_norm' in config.model else False
        fold = config.model.fold if 'fold' in config.model else 1
        cond_channels = config.model.cond_channels if 'cond_channels' in config.model else 0
        cond_fold = config.model.cond_fold if 'cond_fold' in config.model else fold
        
        if config.model.type == 'bayesian':
            self.logvar = self.create_parameter([num_timesteps,], default_initializer=nn.initializer.Constant(0.0))
        
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        
        self.cond_channels = cond_channels
        self.fold = fold
        self.cond_fold = cond_fold

        # timestep embedding
        self.temb = nn.Layer()
        self.temb.dense = nn.LayerList([
            paddle.nn.Linear(self.ch,
                            self.temb_ch),
            paddle.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = paddle.nn.Conv2D(in_channels*fold**2 + cond_channels*cond_fold**2,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.LayerList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.LayerList()
            attn = nn.LayerList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         use_scale_shift_norm=use_scale_shift_norm))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Layer()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Layer()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       use_scale_shift_norm=use_scale_shift_norm)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       use_scale_shift_norm=use_scale_shift_norm)

        # upsampling
        self.up = []
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.LayerList()
            attn = nn.LayerList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         use_scale_shift_norm=use_scale_shift_norm))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Layer()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order
        self.up = nn.LayerList(self.up)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = paddle.nn.Conv2D(block_in,
                                        out_ch*fold**2,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t, c=None):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        # print("#### models -> diffusion -> Model -> forward ####")
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        x = spatial_fold(x, self.fold)

        # downsampling
        if c is not None:
            c = c.astype(x.dtype)
            if len(c.shape) <= 2:
                c = c.reshape([c.shape[0],-1,1,1])
                c = c.tile([1,1,*x.shape[-2:]])
                c = c.tile([1,1,self.cond_fold,self.cond_fold])
            c = spatial_fold(c, self.cond_fold)
            x = paddle.concat([x, c], 1)
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    paddle.concat([h, hs.pop()], 1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = spatial_unfold(h, self.fold)
        return h

class FNO2d(nn.Layer):
    def __init__(self, modes1, modes2, width):
        super(FNO2d,self).__init__()

        self.modes1 = modes1 
        self.modes2 = modes2 
        self.width = width 
        self.fc0 = nn.Linear(1, self.width) 
        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2D(self.width, self.width, 1)
        self.w1 = nn.Conv2D(self.width, self.width, 1)
        self.w2 = nn.Conv2D(self.width, self.width, 1)
        self.w3 = nn.Conv2D(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, n):

        # FC Layer
        x = self.fc0(x)
        x = paddle.transpose(x, perm=[0, 3, 1, 2])

        # Fourier Layers
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

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

        # Inverse Fourier Layers
        x_ft = paddle.fft.rfft2(x)
        x = paddle.fft.irfft2(x_ft, s=(n*x.shape[-2], n*x.shape[-1]))
        x = paddle.transpose(x,perm=[0,2,3,1])
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = paddle.linspace(0, 1, size_x, dtype='float32')
        gridx = gridx.reshape([1, size_x, 1, 1]).repeat([batchsize, 1, size_y, 1])
        gridy = paddle.linspace(0, 1, size_y, dtype='float32')
        gridy = gridy.reshape([1, 1, size_y, 1]).repeat([batchsize, size_x, 1, 1])
        # 使用paddle.concat在最后一个维度上拼接gridx和gridy
        grid = paddle.concat((gridx, gridy), axis=-1)
        return grid

class SpectralConv2d_fast(nn.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2 

        self.weights1_real = self.create_parameter([in_channels, out_channels, self.modes1, self.modes2])
        self.weights2_real = self.create_parameter([in_channels, out_channels, self.modes1, self.modes2])
        self.weights1_img = self.create_parameter([in_channels, out_channels, self.modes1, self.modes2])
        self.weights2_img = self.create_parameter([in_channels, out_channels, self.modes1, self.modes2])
        self.weights1 = paddle.complex(self.weights1_real,self.weights2_img)
        self.weights2 = paddle.complex(self.weights2_real,self.weights2_img)
    

    def compl_mul2d(self, input, weights):
         # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return paddle.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):

        batchsize = x.shape[0]

        x_ft = paddle.fft.rfft2(x)

        out_ft_real = paddle.zeros([batchsize, self.out_channels,  x.shape[-2], x.shape[-1]//2 + 1])
        out_ft_img = paddle.zeros([batchsize, self.out_channels,  x.shape[-2], x.shape[-1]//2 + 1])
        out_ft = paddle.complex(out_ft_real,out_ft_img)

        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = paddle.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))

        return x


class ConditionalModel(nn.Layer):

    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(
            config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps
        if config.model.type == 'bayesian':
            out_9 = paddle.create_parameter(shape=paddle.zeros(shape=
                num_timesteps).shape, dtype=paddle.zeros(shape=
                num_timesteps).numpy().dtype, default_initializer=paddle.nn
                .initializer.Assign(paddle.zeros(shape=num_timesteps)))
            out_9.stop_gradient = not True
            self.logvar = out_9
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.temb = paddle.nn.Layer()
        self.temb.dense = paddle.nn.LayerList(sublayers=[paddle.nn.Linear(
            in_features=self.ch, out_features=self.temb_ch), paddle.nn.
            Linear(in_features=self.temb_ch, out_features=self.temb_ch)])
        self.emb_conv = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=
            in_channels, out_channels=self.ch, kernel_size=1, stride=1,
            padding=0), paddle.nn.GELU(), paddle.nn.Conv2D(in_channels=self
            .ch, out_channels=self.ch, kernel_size=3, stride=1, padding=1,
            padding_mode='circular'))
        self.conv_in = paddle.nn.Conv2D(in_channels=in_channels,
            out_channels=self.ch, kernel_size=3, stride=1, padding=1,
            padding_mode='circular')
        self.combine_conv = paddle.nn.Conv2D(in_channels=self.ch * 2,
            out_channels=self.ch, kernel_size=1, stride=1, padding=0)
        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = paddle.nn.LayerList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = paddle.nn.LayerList()
            attn = paddle.nn.LayerList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels
                    =block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = paddle.nn.Layer()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = paddle.nn.Layer()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=
            block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=
            block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.up = paddle.nn.LayerList()
        for i_level in reversed(range(self.num_resolutions)):
            block = paddle.nn.LayerList()
            attn = paddle.nn.LayerList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                    out_channels=block_out, temb_channels=self.temb_ch,
                    dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = paddle.nn.Layer()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        self.norm_out = Normalize(block_in)
        self.conv_out = paddle.nn.Conv2D(in_channels=block_in, out_channels
            =out_ch, kernel_size=3, stride=1, padding=1, padding_mode=
            'circular')

    def forward(self, x, t, dx=None):
        assert tuple(x.shape)[2] == tuple(x.shape)[3] == self.resolution
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        x = self.conv_in(x)
        if dx is not None:
            cond_emb = self.emb_conv(dx)
        else:
            cond_emb = paddle.zeros_like(x=x)
        x = paddle.concat(x=(x, cond_emb), axis=1)
        hs = [self.combine_conv(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](paddle.concat(x=[h, hs.
                    pop()], axis=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h