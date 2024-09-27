import paddle


def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
>>>>>>        return torch.optim.Adam(parameters, lr=config.optim.lr,
            weight_decay=config.optim.weight_decay, betas=(config.optim.
            beta1, 0.999), amsgrad=config.optim.amsgrad, eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return paddle.optimizer.RMSProp(parameters=parameters,
            learning_rate=config.optim.lr, weight_decay=config.optim.
            weight_decay, epsilon=1e-08, rho=0.99)
    elif config.optim.optimizer == 'SGD':
>>>>>>        return torch.optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(
            config.optim.optimizer))
