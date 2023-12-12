import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate,
                                          betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def step(self, input_train, target_train, input_valid, target_valid, eta,
             network_optimizer, unrolled, patch_mask):
        self.optimizer.zero_grad()

        if unrolled:
            raise NotImplementedError
        else:
            self._backward_step(input_valid, target_valid, patch_mask)

        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid, patch_mask):
        loss = self.model._loss(input_valid, target_valid, patch_mask)
        loss.backward()
