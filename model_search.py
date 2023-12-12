import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
from utils import patchify


class Decoder(nn.Module):
    def __init__(self, C, dataset_stats: tuple, scale_ratio=4):
        super(Decoder, self).__init__()
        self.dataset_min = dataset_stats[0]
        self.dataset_max = dataset_stats[1]

        def upsample(in_feat, out_feat, normalize=True):

            layers = [
                nn.ConvTranspose2d(in_feat,
                                   out_feat,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.ReLU())

            return layers

        self.inter_conv = nn.Sequential(
            nn.Conv2d(C, scale_ratio * C, kernel_size=3, padding=1),
            nn.Conv2d(scale_ratio * C, C, kernel_size=1))
        self.decoder = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1), nn.BatchNorm2d(C),
            nn.ReLU(), *upsample(C, C // 2),
            nn.Conv2d(C // 2, C // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(C // 2), nn.ReLU(), *upsample(C // 2, C // 4),
            nn.Conv2d(C // 4, 3, kernel_size=3, stride=1, padding=1),
            nn.Hardtanh(self.dataset_min, self.dataset_max))

    def forward(self, x):
        x = self.inter_conv(x)
        out = self.decoder(x)
        return out


class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction,
                 reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev,
                                          C,
                                          1,
                                          1,
                                          0,
                                          affine=False)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j])
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self,
                 C,
                 num_classes,
                 layers,
                 criterion,
                 dataset_stats: tuple,
                 steps=4,
                 multiplier=4,
                 stem_multiplier=3,
                 task='cls',
                 patch_size=4):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        if type(criterion) is list:
            self.cls_criterion = criterion[0]
            self.rec_criterion = criterion[1]
        self._steps = steps
        self._multiplier = multiplier
        self.dataset_stats = dataset_stats
        self.task = task
        self.patch_size = (patch_size, patch_size)

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr))

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr,
                        reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        if self.task == 'cls':
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(C_prev, num_classes)
        elif 'rec' in self.task:
            self.classifier = Decoder(C_prev, dataset_stats, scale_ratio=4)
        elif self.task == 'cls_mask':
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(C_prev, num_classes)
            self.rec_decoder = Decoder(C_prev, dataset_stats, scale_ratio=4)

        self._initialize_alphas()

    def forward(self, input):
        inter_feature = []
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            if cell.reduction:
                inter_feature.append(s1.clone().detach().cpu())
            s0, s1 = s1, cell(s0, s1, weights)
        inter_feature.append(s1.clone().detach().cpu())

        if self.task == 'cls':
            out = self.global_pooling(s1)
            logits = self.classifier(out.view(out.size(0), -1))
        elif 'rec' in self.task:
            logits = self.classifier(s1)
        elif self.task == 'cls_mask':
            cls_out = self.global_pooling(s1)
            cls_logits = self.classifier(cls_out.view(cls_out.size(0), -1))
            rec_logits = self.rec_decoder(s1)
            logits = [cls_logits, rec_logits]

        return logits, inter_feature

    def _loss(self, input, target, mask=None):
        logits, _ = self(input)

        if self.task == 'rec' or self.task == 'cls':
            l = self._criterion(logits, target)
        elif self.task == 'mask_rec' and mask is not None:
            logits = patchify(logits, patch_size=self.patch_size)
            l = self._criterion(logits, target, mask)
        elif self.task == 'cls_mask' and mask is not None:
            assert type(logits) is list and type(target) is list

            cls_l = self.cls_criterion(logits[0], target[0])
            rec_logits = patchify(logits[1], patch_size=self.patch_size)
            rec_l = self.rec_criterion(rec_logits, target[1], mask)
            l = cls_l + rec_l / (rec_l / cls_l).detach()

        return l

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(),
                                      requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(),
                                      requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(W[x][k] for k in range(len(W[x]))
                                       if k != PRIMITIVES.index('none')))[:2]

                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))

                start = end
                n += 1

            return gene

        gene_normal = _parse(
            F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(
            F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(normal=gene_normal,
                            normal_concat=concat,
                            reduce=gene_reduce,
                            reduce_concat=concat)
        return genotype
