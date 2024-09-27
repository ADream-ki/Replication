import sys
sys.path.append('D:\\Replication/paddle_project/utils')
import paddle_aux
import paddle
import numpy as np


def voriticity_residual(w, re=1000.0, dt=1 / 32):
    batchsize = w.shape[0]
    w = w.clone()
    out_5 = w
    out_5.stop_gradient = not True
    out_5
    nx = w.shape[2]
    ny = w.shape[3]
    device = w.place
    w_h = paddle.fft.fft2(x=w[:, 1:-1], axes=[2, 3])
    k_max = nx // 2
    N = nx
    k_x = paddle.concat(x=(paddle.arange(start=0, end=k_max, step=1),
        paddle.arange(start=-k_max, end=0, step=1)), axis=0).reshape(N, 1
        ).repeat(1, N).reshape(1, 1, N, N)
    k_y = paddle.concat(x=(paddle.arange(start=0, end=k_max, step=1),
        paddle.arange(start=-k_max, end=0, step=1)), axis=0).reshape(1, N
        ).repeat(N, 1).reshape(1, 1, N, N)
    lap = k_x ** 2 + k_y ** 2
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap
    u_h = 1.0j * k_y * psi_h
    v_h = -1.0j * k_x * psi_h
    wx_h = 1.0j * k_x * w_h
    wy_h = 1.0j * k_y * w_h
    wlap_h = -lap * w_h
    u = paddle.fft.irfft2(x=u_h[..., :, :k_max + 1], axes=[2, 3])
    v = paddle.fft.irfft2(x=v_h[..., :, :k_max + 1], axes=[2, 3])
    wx = paddle.fft.irfft2(x=wx_h[..., :, :k_max + 1], axes=[2, 3])
    wy = paddle.fft.irfft2(x=wy_h[..., :, :k_max + 1], axes=[2, 3])
    wlap = paddle.fft.irfft2(x=wlap_h[..., :, :k_max + 1], axes=[2, 3])
    advection = u * wx + v * wy
    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)
    x = paddle.linspace(start=0, stop=2 * np.pi, num=nx + 1)
    x = x[0:-1]
    X, Y = paddle.meshgrid(x, x)
    f = -4 * paddle.cos(x=4 * Y)
    residual = wt + (advection - 1.0 / re * wlap + 0.1 * w[:, 1:-1]) - f
    residual_loss = (residual ** 2).mean()
    dw = paddle.grad(outputs=residual_loss, inputs=w)[0]
    return dw


def noise_estimation_loss(model, x0: paddle.Tensor, t: paddle.Tensor, e:
    paddle.Tensor, b: paddle.Tensor, keepdim=False):
    a = (1 - b).cumprod(dim=0).index_select(axis=0, index=t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.astype(dtype='float32'))
    if keepdim:
        return (e - output).square().sum(axis=(1, 2, 3))
    else:
        return (e - output).square().sum(axis=(1, 2, 3)).mean(axis=0)


def conditional_noise_estimation_loss(model, x0: paddle.Tensor, t: paddle.
    Tensor, e: paddle.Tensor, b: paddle.Tensor, x_scale, x_offset, keepdim=
    False, p=0.1):
    a = (1 - b).cumprod(dim=0).index_select(axis=0, index=t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    flag = np.random.uniform(0, 1)
    if flag < p:
        output = model(x, t.astype(dtype='float32'))
    else:
        dx = voriticity_residual(x * x_scale + x_offset) / x_scale
        output = model(x, t.astype(dtype='float32'), dx)
    if keepdim:
        return (e - output).square().sum(axis=(1, 2, 3))
    else:
        return (e - output).square().sum(axis=(1, 2, 3)).mean(axis=0)


loss_registry = {'simple': noise_estimation_loss, 'conditional':
    conditional_noise_estimation_loss}
