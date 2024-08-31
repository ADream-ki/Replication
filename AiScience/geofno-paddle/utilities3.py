import numpy as np

import paddle
from paddle import nn, linalg


#loss function with rel/abs Lp loss OK
class LpLoss(nn.Layer):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.shape[0]

        #Assume uniform mesh
        h = 1.0 / (x.shape[1] - 1.0)

        all_norms = (h**(self.d/self.p))*paddle.norm(x.reshape([num_examples,-1]) - y.reshape([num_examples,-1]), self.p, 1)

        if self.reduction:
            if self.size_average:
                return (all_norms).mean()
            else:
                return (all_norms).sum()

        return all_norms
    
    def rel(self, x, y):
        num_examples = x.shape[0] # batch size

        diff_norms = paddle.norm(x.reshape([num_examples,-1]) - y.reshape([num_examples,-1]), self.p, 1)
        y_norms = paddle.norm(y.reshape([num_examples,-1]), self.p, 1)

        if self.reduction:
            if self.size_average:
                return (diff_norms/y_norms).mean()
            else:
                return (diff_norms/y_norms).sum()

        return diff_norms/y_norms
    
    def forward(self, x, y):
        #output  label
        return self.rel(x, y)
    


def count_params(model):
    c = 0
    for p in model.parameters():
        c += np.prod(p.shape)  # 使用np.prod计算参数的总数
    return c