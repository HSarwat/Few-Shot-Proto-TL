import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable
from proto_net.utils.blocks import conv_block, Flatten
from proto_net.utils.dist import euclidean_dist, cosine_dist
from sklearn.metrics import confusion_matrix
import numpy as np


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=32.0, m=0.25, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # if not self.training:
        #    return cosine * self.s
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        cross_entropy_loss = nn.CrossEntropyLoss()(output, label)
        return cross_entropy_loss


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()
        self.encoder = encoder
        self.encoder.apply(init_weights)
        self.arcFace = ArcMarginProduct(48, 10, easy_margin=False)

    @classmethod
    def defualt_encoder(cls, x_dim=1, hid_dim=64, z_dim=64):
        encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            Flatten()
        )
        return cls(encoder)


    @classmethod
    def default_encoder1(cls, input, output):
        encoder = nn.Sequential(
            nn.Linear(in_features=input, out_features=256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, output),
            nn.ReLU(),
        )

        return cls(encoder)

    @classmethod
    def default_encoder2(cls, input, output, ld, l1, l2, l3):
        if ld == 1:
            encoder = nn.Sequential(
                nn.Linear(in_features=input, out_features=l1),
                nn.ReLU(),
                nn.BatchNorm1d(l1),
                nn.Dropout(),

                nn.Linear(l1, output),
            )
        elif ld == 2:
            encoder = nn.Sequential(
                nn.Linear(in_features=input, out_features=l1),
                nn.ReLU(),
                nn.BatchNorm1d(l1),
                nn.Dropout(),

                nn.Linear(l1, l2),
                nn.ReLU(),
                nn.BatchNorm1d(l2),
                nn.Dropout(),

                nn.Linear(l2, output)
            )
        elif ld == 3:
            encoder = nn.Sequential(
                nn.Linear(in_features=input, out_features=l1),
                nn.ReLU(),
                nn.BatchNorm1d(l1),
                nn.Dropout(),

                nn.Linear(l1, l2),
                nn.ReLU(),
                nn.BatchNorm1d(l2),
                nn.Dropout(),

                nn.Linear(l2, l3),
                nn.ReLU(),
                nn.BatchNorm1d(l3),
                nn.Dropout(),

                nn.Linear(l3, l3),
                nn.ReLU(),
                nn.BatchNorm1d(l3),
                nn.Dropout(),
                #
                # nn.Linear(l3, l3),
                # nn.ReLU(),
                # nn.BatchNorm1d(l3),
                # nn.Dropout(),

                nn.Linear(l3, output)
            )

        return cls(encoder)

    def sample_validation(self, sample):

        if 'xs' not in sample or 'xq' not in sample:
            raise ValueError("Protonet loss requires support set 'xs' and query set 'xq'")

        xs = sample['xs']
        xq = sample['xq']

        size = len(xq.size())
        assert size == len(xs.size())

        if size < 3:
            raise ValueError("Error: Data dimension is not in the proper format. \
                             Expected format: [_class, n_query, ...], but received: {size}.")

    def loss(self, sample):
        ''' prtototypical loss

            params:
                sample: dict
                    xs: support set (n_class, n_query, ...)
                    xq: query set (n_class, n_query, ...)
        '''

        self.sample_validation(sample)

        xs = Variable(sample['xs'])  # *0.01
        xq = Variable(sample['xq'])  # *0.01

        n_class = xs.size(0)
        n_support = xs.size(1)
        n_query = xq.size(1)

        assert xq.size(0) == n_class

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.reshape(n_class * n_support, *xs.size()[2:]),
                       xq.reshape(n_class * n_query, *xq.size()[2:])], 0)

        x = F.normalize(x, dim=1)
        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class * n_support:]

        # dists = euclidean_dist(zq, z_proto)
        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        # loss_arc = self.arcFace.forward(zq, target_inds.view(-1))

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        # loss_comb = 0.7 * loss_val + (1.0 - 0.7) * loss_arc

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        y_pred = np.ravel(y_hat.cpu().numpy())
        y_true = np.ravel(target_inds.squeeze().cpu().numpy())
        cm = confusion_matrix(y_true, y_pred)

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'y_true': y_true,
            'y_pred': y_pred,
            'cm': cm
        }

    def predict(self, sample):
        ''' prtototypical prediction

            params:
                sample: dict
                    xs: support set (n_class, n_query, ...)
                    xq: query set (n_query, ...)
        '''

        xs = Variable(sample['xs'])
        xq = Variable(sample['xq'])

        n_class = xs.size(0)
        n_support = xs.size(1)
        n_query = xq.size(0)

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class * n_support:]

        # dists = euclidean_dist(zq, z_proto)
        dists = euclidean_dist(zq, z_proto)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        _, y_hat = log_p_y.max(2)

        return y_hat