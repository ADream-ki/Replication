import paddle
import os
import logging
import time
import glob
import numpy as np
import tqdm
from models.diffusion import Model, ConditionalModel
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from functions.ckpt_util import get_ckpt_path
from tensorboardX import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
paddle.seed(seed=0)
np.random.seed(0)


def torch2hwcuint8(x, clip=False):
    if clip:
        x = paddle.clip(x=x, min=-1, max=1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end,
    num_diffusion_timesteps):

    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5,
            num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps,
            dtype=np.float64)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1,
            num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'sigmoid':
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert tuple(betas.shape) == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):

    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = str('cuda').replace('cuda', 'gpu'
                ) if paddle.device.cuda.device_count() >= 1 else str('cpu'
                ).replace('cuda', 'gpu')
        self.device = device
        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(beta_schedule=config.diffusion.
            beta_schedule, beta_start=config.diffusion.beta_start, beta_end
            =config.diffusion.beta_end, num_diffusion_timesteps=config.
            diffusion.num_diffusion_timesteps)
        betas = self.betas = paddle.to_tensor(data=betas).astype(dtype=
            'float32').to(self.device)
        self.num_timesteps = tuple(betas.shape)[0]
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = paddle.concat(x=[paddle.ones(shape=[1]).to(
            device), alphas_cumprod[:-1]], axis=0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 -
            alphas_cumprod)
        if self.model_var_type == 'fixedlarge':
            self.logvar = betas.log()
        elif self.model_var_type == 'fixedsmall':
            self.logvar = posterior_variance.clip(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        if os.path.exists(config.data.stat_path):
            print('Loading dataset statistics from {}'.format(config.data.
                stat_path))
>>>>>>            train_data = datasets.utils.KMFlowTensorDataset(config.data.
                data_dir, stat_path=config.data.stat_path)
        else:
            print('No dataset statistics found. Computing statistics...')
>>>>>>            train_data = datasets.utils.KMFlowTensorDataset(config.data.
                data_dir)
            train_data.save_data_stats(config.data.stat_path)
        train_loader = paddle.io.DataLoader(dataset=train_data, batch_size=
            config.training.batch_size, shuffle=True, num_workers=config.
            data.num_workers)
        model = Model(config)
        model = model.to(self.device)
        optimizer = get_optimizer(self.config, model.parameters())
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None
        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = paddle.load(path=os.path.join(self.args.log_path,
                'ckpt.pth'))
            model.set_state_dict(state_dict=states[0])
            states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            optimizer.set_state_dict(state_dict=states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.set_state_dict(state_dict=states[4])
        writer = SummaryWriter()
        num_iter = 0
        log_freq = 100
        print('Starting training...')
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            epoch_loss = []
            for i, x in enumerate(train_loader):
                n = x.shape[0]
                data_time += time.time() - data_start
                model.train()
                step += 1
                x = x.to(self.device)
                e = paddle.randn(shape=x.shape, dtype=x.dtype)
                b = self.betas
                t = paddle.randint(low=0, high=self.num_timesteps, shape=(n //
                    2 + 1,)).to(self.device)
                t = paddle.concat(x=[t, self.num_timesteps - t - 1], axis=0)[:n
                    ]
                loss = loss_registry[config.model.type](model, x, t, e, b)
                epoch_loss.append(loss.item())
                tb_logger.add_scalar('loss', loss, global_step=step)
                if num_iter % log_freq == 0:
                    logging.info(
                        f'step: {step}, loss: {loss.item()}, data time: {data_time / (i + 1)}'
                        )
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('data_time', data_time / (i + 1), step)
                """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
>>>>>>                optimizer.zero_grad()
                loss.backward()
                try:
                    paddle.nn.utils.clip_grad_norm_(parameters=model.
                        parameters(), max_norm=config.optim.grad_clip)
                except Exception:
                    pass
                optimizer.step()
                if self.config.model.ema:
                    ema_helper.update(model)
                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [model.state_dict(), optimizer.state_dict(),
                        epoch, step]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())
                    paddle.save(obj=states, path=os.path.join(self.args.
                        log_path, 'ckpt_{}.pth'.format(step)))
                    paddle.save(obj=states, path=os.path.join(self.args.
                        log_path, 'ckpt.pth'))
                data_start = time.time()
                num_iter = num_iter + 1
            print('==========================================================')
            print('Epoch: {}/{}, Loss: {}'.format(epoch, self.config.
                training.n_epochs, np.mean(epoch_loss)))
        print('Finished training')
        logging.info(
            f'step: {step}, loss: {loss.item()}, data time: {data_time / (i + 1)}'
            )
        paddle.save(obj=states, path=os.path.join(self.args.log_path,
            'ckpt_{}.pth'.format(step)))
        paddle.save(obj=states, path=os.path.join(self.args.log_path,
            'ckpt.pth'))
        print('Model saved at: ', self.args.log_path + 'ckpt_{}.pth'.format
            (step))
        writer.export_scalars_to_json('./runs/all_scalars.json')
        writer.close()

    def sample(self):
        pass

    def sample_sequence(self, model):
        config = self.config
        x = paddle.randn(shape=[8, config.data.channels, config.data.
            image_size, config.data.image_size])
        with paddle.no_grad():
            _, x = self.sample_image(x, model, last=False)
>>>>>>        x = [datasets.inverse_data_transform(config, y) for y in x]
        for i in range(len(x)):
            for j in range(x[i].shape[0]):
>>>>>>                torchvision.utils.save_image(x[i][j], os.path.join(self.
                    args.image_folder, f'{j}_{i}.png'))

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = paddle.acos(x=paddle.sum(x=z1 * z2) / (paddle.linalg.
                norm(x=z1) * paddle.linalg.norm(x=z2)))
            return paddle.sin(x=(1 - alpha) * theta) / paddle.sin(x=theta
                ) * z1 + paddle.sin(x=alpha * theta) / paddle.sin(x=theta) * z2
        z1 = paddle.randn(shape=[1, config.data.channels, config.data.
            image_size, config.data.image_size])
        z2 = paddle.randn(shape=[1, config.data.channels, config.data.
            image_size, config.data.image_size])
        alpha = paddle.arange(start=0.0, end=1.01, step=0.1).to(z1.place)
        z_ = []
        for i in range(alpha.shape[0]):
            z_.append(slerp(z1, z2, alpha[i]))
        x = paddle.concat(x=z_, axis=0)
        xs = []
        with paddle.no_grad():
            for i in range(0, x.shape[0], 8):
                xs.append(self.sample_image(x[i:i + 8], model))
>>>>>>        x = datasets.inverse_data_transform(config, paddle.concat(x=xs, axis=0)
            )
        for i in range(x.shape[0]):
>>>>>>            torchvision.utils.save_image(x[i], os.path.join(self.args.
                image_folder, f'{i}.png'))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        if self.args.sample_type == 'generalized':
            if self.args.skip_type == 'uniform':
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == 'quad':
                seq = np.linspace(0, np.sqrt(self.num_timesteps * 0.8),
                    self.args.timesteps) ** 2
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps
            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta
                )
            x = xs
        elif self.args.sample_type == 'ddpm_noisy':
            if self.args.skip_type == 'uniform':
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == 'quad':
                seq = np.linspace(0, np.sqrt(self.num_timesteps * 0.8),
                    self.args.timesteps) ** 2
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps
            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass


class ConditionalDiffusion(object):

    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = str('cuda').replace('cuda', 'gpu'
                ) if paddle.device.cuda.device_count() >= 1 else str('cpu'
                ).replace('cuda', 'gpu')
        self.device = device
        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(beta_schedule=config.diffusion.
            beta_schedule, beta_start=config.diffusion.beta_start, beta_end
            =config.diffusion.beta_end, num_diffusion_timesteps=config.
            diffusion.num_diffusion_timesteps)
        betas = self.betas = paddle.to_tensor(data=betas).astype(dtype=
            'float32').to(self.device)
        self.num_timesteps = tuple(betas.shape)[0]
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = paddle.concat(x=[paddle.ones(shape=[1]).to(
            device), alphas_cumprod[:-1]], axis=0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 -
            alphas_cumprod)
        if self.model_var_type == 'fixedlarge':
            self.logvar = betas.log()
        elif self.model_var_type == 'fixedsmall':
            self.logvar = posterior_variance.clip(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        if os.path.exists(config.data.stat_path):
            print('Loading dataset statistics from {}'.format(config.data.
                stat_path))
>>>>>>            train_data = datasets.utils.KMFlowTensorDataset(config.data.
                data_dir, stat_path=config.data.stat_path)
        else:
            print('No dataset statistics found. Computing statistics...')
>>>>>>            train_data = datasets.utils.KMFlowTensorDataset(config.data.
                data_dir)
            train_data.save_data_stats(config.data.stat_path)
        x_offset, x_scale = train_data.stat['mean'], train_data.stat['scale']
        train_loader = paddle.io.DataLoader(dataset=train_data, batch_size=
            config.training.batch_size, shuffle=True, num_workers=config.
            data.num_workers)
        model = ConditionalModel(config)
        num_params = sum(p.size for p in model.parameters() if not p.
            stop_gradient)
        print(num_params)
        model = model.to(self.device)
        optimizer = get_optimizer(self.config, model.parameters())
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None
        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = paddle.load(path=os.path.join(self.args.log_path,
                'ckpt.pth'))
            model.set_state_dict(state_dict=states[0])
            states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            optimizer.set_state_dict(state_dict=states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.set_state_dict(state_dict=states[4])
        writer = SummaryWriter()
        num_iter = 0
        log_freq = 100
        print('Starting training...')
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            epoch_loss = []
            for i, x in enumerate(train_loader):
                n = x.shape[0]
                data_time += time.time() - data_start
                model.train()
                step += 1
                x = x.to(self.device)
                e = paddle.randn(shape=x.shape, dtype=x.dtype)
                b = self.betas
                t = paddle.randint(low=0, high=self.num_timesteps, shape=(n //
                    2 + 1,)).to(self.device)
                t = paddle.concat(x=[t, self.num_timesteps - t - 1], axis=0)[:n
                    ]
                loss = loss_registry[config.model.type](model, x, t, e, b,
                    x_offset.item(), x_scale.item())
                epoch_loss.append(loss.item())
                tb_logger.add_scalar('loss', loss, global_step=step)
                if num_iter % log_freq == 0:
                    logging.info(
                        f'step: {step}, loss: {loss.item()}, data time: {data_time / (i + 1)}'
                        )
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('data_time', data_time / (i + 1), step)
                """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
>>>>>>                optimizer.zero_grad()
                loss.backward()
                try:
                    paddle.nn.utils.clip_grad_norm_(parameters=model.
                        parameters(), max_norm=config.optim.grad_clip)
                except Exception:
                    pass
                optimizer.step()
                if self.config.model.ema:
                    ema_helper.update(model)
                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [model.state_dict(), optimizer.state_dict(),
                        epoch, step]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())
                    paddle.save(obj=states, path=os.path.join(self.args.
                        log_path, 'ckpt_{}.pth'.format(step)))
                    paddle.save(obj=states, path=os.path.join(self.args.
                        log_path, 'ckpt.pth'))
                data_start = time.time()
                num_iter = num_iter + 1
            print('==========================================================')
            print('Epoch: {}/{}, Loss: {}'.format(epoch, self.config.
                training.n_epochs, np.mean(epoch_loss)))
        print('Finished training')
        logging.info(
            f'step: {step}, loss: {loss.item()}, data time: {data_time / (i + 1)}'
            )
        paddle.save(obj=states, path=os.path.join(self.args.log_path,
            'ckpt_{}.pth'.format(step)))
        paddle.save(obj=states, path=os.path.join(self.args.log_path,
            'ckpt.pth'))
        print('Model saved at: ', self.args.log_path + 'ckpt_{}.pth'.format
            (step))
        writer.export_scalars_to_json('./runs/all_scalars.json')
        writer.close()

    def sample(self):
        pass

    def sample_sequence(self, model):
        pass

    def sample_interpolation(self, model):
        pass

    def sample_image(self, x, model, last=True):
        pass

    def test(self):
        pass
