
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import Config
from losses import *
from modules import Generator, Discriminator


opt = Config()

dataset = datasets.CIFAR10(
    root='./.data',
    train = True,
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5,0.5)]),
    download = True
)
train_loader = DataLoader(
    dataset = dataset, batch_size= opt.batch_size, shuffle =True
)

if opt.device == 'cpu':
    DEVICE = torch.device('cpu')
else :
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

discriminator = Discriminator(opt.dis_feature_dim, DEVICE).to(DEVICE) # Discriminator 초기화 하면서 pivotal vector를 초기화 시켜준다.
generator = Generator(opt.hid_dim, opt.gen_feature_dim).to(DEVICE)

# 처음 center 초기화.
c = torch.Tensor(torch.zeros(1, opt.dis_feature_dim)).to(DEVICE)

c_optim = torch.optim.Adam(params = [c], lr =opt.lr, betas =opt.betas)
dis_optim = torch.optim.Adam(params = discriminator.parameters(), lr =opt.lr, betas =opt.betas)
gen_optim = torch.optim.Adam(params = generator.parameters(), lr =opt.lr, betas =opt.betas)



"""
    Dataset : CIFAR 10
"""
EPOCHS = opt.EPOCHS

for epoch in range(EPOCHS):
    print("EPOCH : {}/{}".format(epoch + 1, 100))
    real_scores = []
    div_scores = []
    center_losses = []
    radius_losses = []
    for img, _ in tqdm(train_loader):
        img = img.to(DEVICE)
        bs = img.size(0)
        latent = torch.randn(bs, 128).to(DEVICE)

        fake_img = generator(latent).detach()

        embed_vec, _ = discriminator(torch.cat([img, fake_img], dim=0))
        # Batch_size*2, embed_dim 나중에 인덱싱으로 절반 앞부분은 real, 절반 뒷부분은 fake으로 빼오면 된다.

        """
            Center Update
        """
        c_loss = center_loss(embed_vec, c)
        center_losses.append(c_loss.item())
        # c를 update 해주는 방안을 생각해보자.
        c_optim.zero_grad()
        c_loss.backward()
        c_optim.step()

        """
            Discriminator Train
        """
        embed_vec, pivot = discriminator(torch.cat([img, fake_img], dim=0))
        pivot_norm = F.normalize(pivot, p=2.0, dim=1, eps=1e-12, out=None) # pivot을 normalize 해줘야한다.
        
        v = embed_vec - c
        radius = safe_norm(v, dim =1)
        pred_radius = torch.sqrt(torch.mean(torch.square(radius)))
        dis_radius_loss = radius_criterion(radius, torch.ones_like(radius)*pred_radius)

        v = F.normalize(v, p=2.0, dim=1, eps=1e-12, out=None)
        v_proj = pivot_norm * torch.sum(pivot_norm * v, dim =1, keepdims = True)
        v_rej = v - v_proj
        norm_proj = safe_norm(v_proj,dim=1, keepdims=True)
        norm_rej = safe_norm(v_rej,dim=1, keepdims=True)
        sigma_rej = safe_sigma(norm_rej)
        sigma_proj = safe_sigma(norm_proj)

        realness = norm_proj / sigma_proj
        real_scores.append(realness.cpu().detach().numpy())
        diversity = norm_rej / sigma_rej
        div_scores.append(diversity.cpu().detach().numpy())

        theta = torch.arctan(diversity/realness)
        real_theta, fake_theta = torch.split(theta, [bs, bs])
        
        dis_adv_loss = get_adv_loss(real_theta, fake_theta)
        dis_gan_loss = dis_adv_loss + dis_radius_loss

        dis_optim.zero_grad()
        dis_gan_loss.backward()
        dis_optim.step()

        """
            Generator Train
        """
        latent = torch.randn(bs, 128).to(DEVICE)

        fake_img = generator(latent)
        embed_vec, pivot = discriminator(torch.cat([img, fake_img]))
        pivot_norm = F.normalize(pivot, p=2.0, dim=1, eps=1e-12, out=None)
        v = embed_vec - c
        v = F.normalize(v, p=2.0, dim=1, eps=1e-12, out=None)
        v_proj = pivot_norm * torch.sum(pivot_norm * v, dim =1, keepdims = True)
        v_rej = v - v_proj
        norm_proj = safe_norm(v_proj,dim=1, keepdims=True)
        norm_rej = safe_norm(v_rej,dim=1, keepdims=True)
        sigma_rej = safe_sigma(norm_rej)
        sigma_proj = safe_sigma(norm_proj)

        realness = norm_proj / sigma_proj
        diversity = norm_rej / sigma_rej

        real_scores.append(realness.cpu().detach().numpy())
        div_scores.append(diversity.cpu().detach().numpy())

        theta = torch.arctan(diversity/realness)
        real_theta, fake_theta = torch.split(theta, [bs, bs])
        
        gen_gan_loss = get_adv_loss(fake_theta, real_theta)

        gen_optim.zero_grad()
        gen_gan_loss.backward()
        gen_optim.step()
    print("Real Score : {}, Diveristy Score : {}".format(np.mean(real_scores),np.mean(div_scores)))



