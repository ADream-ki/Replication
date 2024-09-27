import sys
sys.path.append('D:\\Replication/paddle_project/utils')
import paddle_aux
import paddle


def compute_alpha(beta, t):
    beta = paddle.concat(x=[paddle.zeros(shape=[1]).to(beta.place), beta],
        axis=0)
    a = (1 - beta).cumprod(dim=0).index_select(axis=0, index=t + 1).view(-1,
        1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, **kwargs):
    with paddle.no_grad():
        n = x.shape[0]
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (paddle.ones(shape=n) * i).to(x.place)
            next_t = (paddle.ones(shape=n) * j).to(x.place)
            at = compute_alpha(b, t.astype(dtype='int64'))
            at_next = compute_alpha(b, next_t.astype(dtype='int64'))
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = kwargs.get('eta', 0) * ((1 - at / at_next) * (1 - at_next) /
                (1 - at)).sqrt()
            c2 = (1 - at_next - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * paddle.randn(shape=x.
                shape, dtype=x.dtype) + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with paddle.no_grad():
        n = x.shape[0]
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (paddle.ones(shape=n) * i).to(x.place)
            next_t = (paddle.ones(shape=n) * j).to(x.place)
            at = compute_alpha(betas, t.astype(dtype='int64'))
            atm1 = compute_alpha(betas, next_t.astype(dtype='int64'))
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')
            output = model(x, t.astype(dtype='float32'))
            e = output
            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = paddle.clip(x=x0_from_e, min=-1, max=1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (atm1.sqrt() * beta_t * x0_from_e + (1 - beta_t).
                sqrt() * (1 - atm1) * x) / (1.0 - at)
            mean = mean_eps
            noise = paddle.randn(shape=x.shape, dtype=x.dtype)
            mask = 1 - (t == 0).astype(dtype='float32')
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * paddle.exp(x=0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
