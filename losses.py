import torch

def real_score(emb_vec, pivot, c):
    """
        emb_vec : bs, emb_dim
        pivot : bs, emb_dim
        c : bs, emb_dim
    """
    bs = emb_vec.size(0)

    v = (emb_vec - c) / torch.linalg.norm(emb_vec - c, dim=1).unsqueeze(1)

    v_proj = torch.tensordot(v.unsqueeze(1), pivot.unsqueeze(2).repeat(bs, 1, 1), dims=2) * pivot  # 32 by emb_dim

    sigma = torch.sqrt(torch.sum(torch.linalg.norm(v_proj, dim=1) ** 2))
    return -torch.linalg.norm(v_proj, dim=1) / sigma, v, v_proj


def div_score(v_proj, v):
    bs = v_proj.size(0)
    v_rej = v - v_proj
    sigma = torch.sqrt(torch.sum(torch.linalg.norm(v_rej, dim=1) ** 2))
    return torch.linalg.norm(v_rej, dim=1) / sigma


def mult_score(real_score, div_score, factor=10.):
    return factor * torch.arctan(div_score / (-1 * real_score))  # [bs, 1]


"""
    여기 확인 해봐야할듯. generator때는 어떻게 되는지. relativistic GAN
"""


def get_adv_loss(real_mult_score, gen_mult_score, inverse=False):  # inverse -> generator 훈련.
    loss = -torch.mean(torch.log(torch.sigmoid(real_mult_score - torch.mean(gen_mult_score)))) - torch.mean(
        torch.log(torch.sigmoid(torch.mean(real_mult_score) - gen_mult_score)))
    if inverse:
        return 1 / loss
    return loss


def my_huber_loss(loss):  # loss는 bs, 1 형식의 텐서
    mask = loss <= 1.
    return torch.mean(torch.where(mask, 0.5 * loss ** 2, loss - 0.5))


def center_loss(v_embed, c):  # bs, embed_dim
    return my_huber_loss(torch.linalg.norm(v_embed- c,  dim=1))


def radius_eq_loss(v, v_emb, c):
    bs = v.size(0)

    R = torch.sqrt(torch.mean(torch.linalg.norm(v - c) ** 2))

    loss = my_huber_loss(R - torch.linalg.norm(v_emb - c))
    return loss / bs


