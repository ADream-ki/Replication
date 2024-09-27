import paddle
import numpy as np

def voriticity_residual(w, re=1000.0, dt=1/32):
    # w [b t h w]
    batchsize = w.shape[0]
    w = w.clone()
    w.stop_gradient = False
    nx = w.shape[2]
    ny = w.shape[3]
    device = w.place

    w_h = paddle.fft.fft2(w[:, 1:-1], axis=[2, 3])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = paddle.concat([
        paddle.arange(start=0, end=k_max, step=1, dtype='float32', place=device),
        paddle.arange(start=-k_max, end=0, step=1, dtype='float32', place=device)
    ], axis=0).reshape([N, 1]).tile([1, N]).reshape([1, 1, N, N])
    k_y = paddle.concat([
        paddle.arange(start=0, end=k_max, step=1, dtype='float32', place=device).reshape([1, N]),
        paddle.arange(start=-k_max, end=0, step=1, dtype='float32', place=device).reshape([1, N])
    ], axis=0).tile([N, 1]).reshape([1, 1, N, N])
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap

    u_h = 1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    u = paddle.fft.irfft2(u_h[..., :, :k_max + 1], axis=[2, 3])
    v = paddle.fft.irfft2(v_h[..., :, :k_max + 1], axis=[2, 3])
    wx = paddle.fft.irfft2(wx_h[..., :, :k_max + 1], axis=[2, 3])
    wy = paddle.fft.irfft2(wy_h[..., :, :k_max + 1], axis=[2, 3])
    wlap = paddle.fft.irfft2(wlap_h[..., :, :k_max + 1], axis=[2, 3])
    advection = u * wx + v * wy

    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)

    # establish forcing term
    x = paddle.linspace(0, 2*np.pi, nx + 1, dtype='float32', place=device)
    x = x[:-1]
    X, Y = paddle.meshgrid(x, x)
    f = -4 * paddle.cos(4 * Y)

    residual = wt + (advection - (1.0 / re) * wlap + 0.1 * w[:, 1:-1]) - f
    residual_loss = paddle.mean(residual ** 2)
    dw = paddle.grad(residual_loss, w)[0]
    return dw

def noise_estimation_loss(model,
                          x0: paddle.Tensor,
                          t: paddle.Tensor,
                          e: paddle.Tensor,
                          b: paddle.Tensor, keepdim=False):
    a = (1-b).cumprod(0).index_select(t, 0).reshape((-1, 1, 1, 1))
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.astype('float32'))
    if keepdim:
        return (e - output).square().sum((1, 2, 3))
    else:
        return (e - output).square().sum((1, 2, 3)).mean(0)

def conditional_noise_estimation_loss(model,
                          x0: paddle.Tensor,
                          t: paddle.LongTensor,
                          e: paddle.Tensor,
                          b: paddle.Tensor,
                          x_scale,
                          x_offset,
                          keepdim=False, p=0.1):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()

    flag = np.random.uniform(0, 1)
    if flag < p:
        output = model(x, t.astype(dtype='float32'))
    else:
        dx = voriticity_residual((x*x_scale + x_offset)) / x_scale
        output = model(x, t.astype(dtype='float32'), dx)
    if keepdim:
        return (e - output).square().sum(axis=(1, 2, 3))
    else:
        return (e - output).square().sum(axis=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
    'conditional': conditional_noise_estimation_loss
}
