
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

for epoch in range(EPOCH):
    print("EPOCH : {}/{}".format(epoch + 1, EPOCH))
    real_scores = []
    div_scores = []
    center_losses = []
    radius_losses = []
    for img, _ in tqdm(train_loader):
        img = img.to(DEVICE)

        latent = torch.randn(opt.batch_size, opt.hid_dim).to(DEVICE)

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
        pivot = F.normalize(pivot, p=2.0, dim=1, eps=1e-12, out=None) # pivot을 normalize 해줘야한다.
        real_s, v, v_proj = real_score(embed_vec, pivot, c)
        real_scores.append(real_s.cpu().detach().numpy())
        div_s = div_score(v_proj, v)
        div_scores.append(div_s.cpu().detach().numpy())
        mult = mult_score(real_s, div_s, factor=10.)

        dis_adv_loss = get_adv_loss(mult[:opt.batch_size], mult[opt.batch_size:], inverse=False)
        radius_loss = radius_eq_loss(embed_vec, c)
        radius_losses.append(radius_loss.cpu().detach().numpy())
        dis_loss = dis_adv_loss + radius_loss

        dis_optim.zero_grad()
        dis_loss.backward()
        dis_optim.step()

        """
            Generator Train
        """
        latent = torch.randn(opt.batch_size, opt.hid_dim).to(DEVICE)

        fake_img = generator(latent)
        embed_vec, pivot = discriminator(torch.cat([img, fake_img]))
        r_score, v, v_proj = real_score(embed_vec, pivot, c)
        real_scores.append(r_score.cpu().detach().numpy())
        d_score = div_score(v_proj, v)
        div_scores.append(d_score.cpu().detach().numpy())

        mult = mult_score(r_score, d_score, factor=10.)
        dis_adv_loss = get_adv_loss(mult[:opt.batch_size], mult[opt.batch_size:], inverse=True)
        
        gen_optim.zero_grad()
        dis_adv_loss.backward()
        gen_optim.step()

    print("Real Score : {}, Diveristy Score : {}, Center Loss : {}, Radius Loss : {}".format(np.mean(real_scores),
                                                                                             np.mean(div_scores),
                                                                                             np.mean(center_losses),
                                                                                             np.mean(radius_losses)))




