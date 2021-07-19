import torch
import torch.nn as nn

def get_adv_loss(logits1, logits2): # inverse -> generator 훈련.
    logits1 = 5.0 * logits1
    logits2 = 5.0 * logits2
    logits1_mean = torch.mean(logits1, dim=0, keepdims=True)
    logits2_mean = torch.mean(logits2, dim=0, keepdims=True)
    loss_rgt = -torch.mean(torch.log(torch.sigmoid(logits1_mean - logits2)+1e-8))
    loss_inv = -torch.mean(torch.log(torch.sigmoid(logits1 - logits2_mean)+1e-8))
    return loss_rgt + loss_inv


def center_loss(v_embed, c): # bs, embed_dim

    criterion = nn.HuberLoss()
    norm = torch.linalg.norm(v_embed - c, dim=1)
    loss = criterion(norm, torch.zeros_like(norm))
    return loss

def safe_norm(x, dim=None, keepdims=False, eps = 1e-10):
    return torch.sqrt(torch.sum(x**2, dim= dim, keepdims=keepdims)+eps)

def safe_sigma(x, eps = 1e-10):
    return torch.sqrt(torch.mean(torch.square(x)+eps))

