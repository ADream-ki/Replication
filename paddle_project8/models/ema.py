import paddle


class EMAHelper(object):

    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, paddle.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if not param.stop_gradient:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, paddle.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if not param.stop_gradient:
                self.shadow[name].data = (1.0 - self.mu
                    ) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, paddle.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if not param.stop_gradient:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, paddle.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(
                inner_module.config.device)
            module_copy.set_state_dict(state_dict=inner_module.state_dict())
            module_copy = paddle.DataParallel(layers=module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.set_state_dict(state_dict=module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
