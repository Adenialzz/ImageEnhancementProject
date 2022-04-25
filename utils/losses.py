import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def single_emd_loss(p, q, r=2):
    """
    Earth Mover's Distance of one sample

    Args:
        p: true distribution of shape num_classes × 1
        q: estimated distribution of shape num_classes × 1
        r: norm parameter
    """
    assert p.shape == q.shape, "Length of the two distribution must be the same"
    length = p.shape[0]
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += torch.abs(sum(p[:i] - q[:i])) ** r
    return (emd_loss / length) ** (1. / r)

def emd_loss(p, q, r=2):
    """
    Earth Mover's Distance on a batch

    Args:
        p: true distribution of shape mini_batch_size × num_classes × 1
        q: estimated distribution of shape mini_batch_size × num_classes × 1
        r: norm parameters
    """
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_emd_loss(p[i], q[i], r=r))
    return sum(loss_vector) / mini_batch_size

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negtive):
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negtive).pow(2).sum(1)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
       
'''
class ContrastiveLoss(nn.Module):
    def __init__(self, T=0.5):
        super(ContrastiveLoss, self).__init__()
        self.T = T
    
    def forward(self, feat, label):
        bs = feat.shape[0]
        similarity_matrix = F.cosine_similarity(feat.unsqueeze(1), feat.unsqueeze(0), dim=2)
        mask = torch.ones_like(similarity_matrix) * (label.expand(bs, bs).eq(label.expand(bs, bs).t()))
        mask_no_sim = torch.ones_like(mask) - mask
        mask_eye = torch.ones(bs, bs) - torch.eye(bs, bs)
        similarity_matrix = torch.exp(similarity_matrix / self.T)
        similarity_matrix *= mask_eye
        sim = mask * similarity_matrix
        no_sim = similarity_matrix - sim
        no_sim_sum = torch.sum(no_sim, dim=1)
        no_sim_sum_expend = no_sim_sum.repeat(bs, 1).T
        sim_sum = sim + no_sim_sum_expend
        loss = torch.div(sim, sim_sum)
        loss = mask_no_sim + loss + torch.eye(bs, bs)
        loss = -torch.log(loss)
        loss = torch.sum(torch.sum(loss, dim=1)) / (2 * bs)
        return loss
'''

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))			# 超参数 温度
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())		# 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):		# emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs

        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


if __name__ == "__main__":
    bs = 8
    loss_func = ContrastiveLoss(bs, device='cpu')
    l1 = torch.zeros(bs, 512)
    l2 = torch.zeros(bs, 512)
    print(loss_func(l1, l2))

