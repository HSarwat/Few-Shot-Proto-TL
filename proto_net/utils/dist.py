import torch
import torch.nn.functional as F


def euclidean_dist(x: torch.Tensor, y: torch.Tensor):
    ''' Compute euclidean distance between two tensors

        params:
            x: n x d
            y: m x d
    '''

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def cosine_dist(x: torch.Tensor, y: torch.Tensor, eps=1e-6, scale=64):
     ''' Compute similarity distance between two tensors

         paramas:
             x: n x d
             y: m x d
     '''

     n = x.size(0)
     m = y.size(0)
     d = x.size(1)

     assert d == y.size(1)

     x_norm = F.normalize(x, dim=1)
     y_norm = F.normalize(y, dim=1)

     x_norm = x_norm.unsqueeze(1).expand(n, m, d)
     y_norm = y_norm.unsqueeze(0).expand(n, m, d)

     dist = (1.0 + eps) - F.cosine_similarity(x_norm, y_norm, dim=2)
     return dist * scale