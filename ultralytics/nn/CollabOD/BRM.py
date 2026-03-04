import torch
import torch.nn as nn
from ..modules import Conv

__all__ = ['BRM']

class BRM(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()
        
        self.conv_align1 = Conv(inc[0], ouc, 1)
        self.conv_align2 = Conv(inc[1], ouc, 1)
        
        self.conv_concat = Conv(ouc * 2, ouc * 2, 3)
        self.sigmoid = nn.Sigmoid()
        
        self.x1_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        self.x2_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        
        self.conv_final = Conv(ouc, ouc, 1)
        
    def forward(self, x):
        self._clamp_abs(self.x1_param.data, 1.0)
        self._clamp_abs(self.x2_param.data, 1.0)
        
        x1, x2 = x
        x1, x2 = self.conv_align1(x1), self.conv_align2(x2)
        x_concat = self.sigmoid(self.conv_concat(torch.cat([x1, x2], dim=1)))
        x1_weight, x2_weight = torch.chunk(x_concat, 2, dim=1)
        x1, x2 = x1 * x1_weight, x2 * x2_weight
        
        return self.conv_final(x1 * self.x1_param + x2 * self.x2_param)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign=data.sign()
            data.abs_().clamp_(value)
            data*=sign
    