import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    Least Squares GAN Loss (LSGAN) — used in CycleGAN paper.
    
    More stable than binary cross-entropy during training.
    
    For Generator : tries to make discriminator output = 1 (real)
    For Discriminator : real images = 1, fake images = 0
    """

    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, prediction, is_real):
        if is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        return self.loss(prediction, target)


class CycleLoss(nn.Module):
    """
    Cycle Consistency Loss — core idea of CycleGAN.
    
    Forward : x -> G(x) -> F(G(x)) ~ x
    Backward: y -> F(y) -> G(F(y)) ~ y
    
    Uses L1 loss (paper section 3.2).
    Weighted by lambda=10 in full objective.
    """

    def __init__(self):
        super(CycleLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, reconstructed, real):
        return self.loss(reconstructed, real)


class IdentityLoss(nn.Module):
    """
    Identity Loss — helps preserve color composition.
    
    If you feed a DSLR image into the DSLR generator,
    it should return the same image (no change needed).
    
    G(y) ~ y  and  F(x) ~ x
    
    Weighted by lambda * 0.5 in full objective.
    Especially useful for photo <-> painting tasks.
    """

    def __init__(self):
        super(IdentityLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, identity_output, real):
        return self.loss(identity_output, real)