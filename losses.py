import torch
import torch.nn as nn

def real_score(emb_vec, pivot, c):
    """
        emb_vec : bs, emb_dim
        pivot : bs, emb_dim
        c : bs, emb_dim
    """
    bs = emb_vec.size(0)
    
    v = (emb_vec - c)/torch.linalg.norm(emb_vec-c, dim=1).unsqueeze(1)


    v_proj = torch.tensordot(v.unsqueeze(1),pivot.unsqueeze(2).repeat(bs,1,1), dims= 2) * pivot  # 32 by emb_dim

    sigma = torch.sqrt(torch.sum(torch.linalg.norm(v_proj, dim=1)**2))
    return -torch.linalg.norm(v_proj,dim=1)/sigma, v, v_proj

def div_score(v_proj, v):
    bs = v_proj.size(0)
    v_rej = v- v_proj
    sigma = torch.sqrt(torch.sum(torch.linalg.norm(v_rej, dim=1)**2))
    return torch.linalg.norm(v_rej, dim=1)/sigma

def mult_score(r_score, d_score, factor=10.):
    return factor*torch.arctan(d_score/(-1*r_score)) # [bs, 1]

"""
    여기 확인 해봐야할듯. generator때는 어떻게 되는지. relativistic GAN
"""
def get_adv_loss(real_mult_score, gen_mult_score): # inverse -> generator 훈련.
    loss = -torch.mean(torch.log(torch.sigmoid(real_mult_score - torch.mean(gen_mult_score)+1e-8)))-torch.mean(torch.log(torch.sigmoid(torch.mean(real_mult_score) - gen_mult_score)+1e-8))
    
    return loss

# def my_huber_loss(loss): # loss는 bs, 1 형식의 텐서
#     mask = torch.abs(loss) <= 1.

#     return torch.mean(torch.where(mask, 0.5*loss**2, loss - 0.5))
    


def center_loss(v_embed, c): # bs, embed_dim
    # bs = v_embed.size(0)
    
    # return my_huber_loss(torch.linalg.norm(v_embed - c, dim=1))/bs
    criterion = nn.HuberLoss()
    norm = torch.linalg.norm(v_embed - c, dim=1)
    loss = criterion(norm, torch.zeros_like(norm))
    
    return loss



def radius_eq_loss(v_emb, c):
    # R = torch.sqrt(torch.mean(torch.linalg.norm(v-c, dim = 1)**2))

    # loss = my_huber_loss(R-torch.linalg.norm(v_emb - c, dim = 1))
    # return loss/(bs**2)
    criterion = nn.HuberLoss()
    norm = torch.linalg.norm(v_emb - c, dim = 1)
    R = torch.sqrt(torch.mean(torch.linalg.norm(v_emb-c, dim = 1)**2, dim=-1, keepdim = True))
    loss = criterion(R.expand_as(norm), norm)
    # 원래 논문상에서는 loss / bs 여야 하는데, 코드에서는 그냥 Loss로 쓴다..
    return loss

