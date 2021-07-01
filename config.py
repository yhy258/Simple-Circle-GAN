class Config:
    batch_size = 1
    EPOCHS = 100
    lr = 1e-4
    betas = [0.5, 0.999]
    dis_feature_dim = 128
    gen_feature_dim = 256
    hid_dim = 150

    device = 'cpu'
