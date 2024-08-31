import paddle
from paddle import nn, linalg


#loss function with rel/abs Lp loss
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
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*linalg.norm(x.reshape([num_examples,-1]) - y.reshape([num_examples,-1]), self.p, 1)

        if self.reduction:
            if self.size_average:
                return paddle.mean(all_norms)
            else:
                return paddle.sum(all_norms)

        return all_norms
    
    def rel(self, x, y):
        num_examples = x.shape[0]

        diff_norms = linalg.norm(x.reshape([num_examples,-1]) - y.reshape([num_examples,-1]), self.p, 1)
        y_norms = linalg.norm(y.reshape([num_examples,-1]), self.p, 1)

        if self.reduction:
            if self.size_average:
                return paddle.mean(diff_norms/y_norms)
            else:
                return paddle.sum(diff_norms/y_norms)

        return diff_norms/y_norms
    
    def forward(self, x, y):
        return self.rel(x, y)