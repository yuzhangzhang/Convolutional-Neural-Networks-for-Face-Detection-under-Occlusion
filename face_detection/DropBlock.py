import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1


class DropBlock2D(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=7):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # Set gamma, set it to 1 if it is smaller than gamma, and set it to 0 if it is greater than gamma
            # This calculation can get the number of random points of the discarded ratio
            gamma = self.drop_prob / (self.block_size ** 2)
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            # print('mask:{}'.format(type(mask)))
            # mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)
            # apply block mask
            out = x * block_mask[:, None, :, :]
            # Normalize the features
            out = out * block_mask.numel() / block_mask.sum()
            # print(out)
            return out

    def _compute_block_mask(self, mask):
        # Take the maximum value, so that can take out 1 of the block size of a block as a drop.
        # Of course, need to flip the size so that 1 is 0 and 0 is 1.
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size,
                                               self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)
        return block_mask