import torch
from torch import nn


class Cauchy_Loss(nn.Module):

    def __init__(self, scale):
        super(Cauchy_Loss, self).__init__()
        self.scale = scale

    def forward(self, input):
        '''
        Our own implementation of the couchy loss function (aka Lorentzian)
        Reference: Jonathan T. Barron, "A General and Adaptive Robust Loss Function", CVPR, 2019
        Args:
            x: "The residual for which the loss is being computed. x can have any shape,
                and alpha and scale will be broadcasted to match x's shape if necessary.
                Must be a tensor of floats."
            scale: "The scale parameter of the loss. When |x| < scale, the loss is an
                L2-like quadratic bowl, and when |x| > scale the loss function takes on a
                different shape according to alpha. Must be a tensor of single-precision
                floats."
        Returns:
            loss value
        '''

        loss = self.log1p_safe(0.5 * (input / self.scale) ** 2)

        return loss

    def log1p_safe(self, x):
        '''
        The same as torch.log1p(x), but clamps the input to prevent NaNs.
        source: https://github.com/jonbarron/robust_loss_pytorch/blob/master/robust_loss_pytorch/util.py
        '''
        # x = torch.as_tensor(x)
        return torch.log1p(torch.min(x, torch.tensor(33e37).to(x)))