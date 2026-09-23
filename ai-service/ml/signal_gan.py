from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class SignalGenerator(nn.Module):
    def __init__(self, noise_dim=64, channels=3, length=256):
        super().__init__(); self.length=length
        self.fc=nn.Linear(noise_dim,128*(length//16))
        self.net=nn.Sequential(
            nn.ConvTranspose1d(128,96,4,2,1),nn.GELU(),
            nn.ConvTranspose1d(96,48,4,2,1),nn.GELU(),
            nn.ConvTranspose1d(48,channels,4,2,1),nn.Tanh())
    def forward(self,z):
        x=self.fc(z).view(z.size(0),128,self.length//16); return self.net(x)

class SignalDiscriminator(nn.Module):
    def __init__(self,channels=3):
        super().__init__(); self.net=nn.Sequential(
            nn.Conv1d(channels,48,7,2,3),nn.LeakyReLU(.2),
            nn.Conv1d(48,96,5,2,2),nn.LeakyReLU(.2),
            nn.Conv1d(96,128,3,2,1),nn.LeakyReLU(.2),nn.AdaptiveAvgPool1d(1))
        self.head=nn.Linear(128,1)
    def forward(self,x): return self.head(self.net(x).squeeze(-1)).squeeze(-1)

def train_signal_gan(output='/cache/clinevo-owned/signal-gan.pt',epochs=10,batch_size=32,seed=7):
    rng=np.random.default_rng(seed); torch.manual_seed(seed)
    t=np.linspace(0,4,256,dtype=np.float32)
    samples=[]
    for _ in range(1024):
        f=float(rng.uniform(.3,3.0)); x=np.stack([np.sin(2*np.pi*f*t),np.cos(2*np.pi*f*t),0.5*np.sin(2*np.pi*f*t+0.3)],axis=0)
        x += rng.normal(0,.05,x.shape); samples.append(x.astype(np.float32))
    loader=DataLoader(TensorDataset(torch.tensor(np.stack(samples))),batch_size=batch_size,shuffle=True)
    G=SignalGenerator(); D=SignalDiscriminator(); gopt=torch.optim.AdamW(G.parameters(),lr=2e-4,betas=(.5,.999)); dopt=torch.optim.AdamW(D.parameters(),lr=2e-4,betas=(.5,.999)); loss=nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        for (real,) in loader:
            b=real.size(0); ones=torch.ones(b); zeros=torch.zeros(b)
            z=torch.randn(b,64); fake=G(z).detach(); dloss=loss(D(real),ones)+loss(D(fake),zeros); dopt.zero_grad(); dloss.backward(); dopt.step()
            z=torch.randn(b,64); fake=G(z); gloss=loss(D(fake),ones); gopt.zero_grad(); gloss.backward(); gopt.step()
        print(f'epoch={epoch+1} d={float(dloss):.4f} g={float(gloss):.4f}')
    path=Path(output); path.parent.mkdir(parents=True,exist_ok=True); torch.save({'generator':G.state_dict(),'discriminator':D.state_dict(),'seed':seed,'architecture':'ClinevoSignalGAN-v1'},path); print(path)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',default='/cache/clinevo-owned/signal-gan.pt'); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--seed',type=int,default=7); a=p.parse_args(); train_signal_gan(a.output,a.epochs,a.batch_size,a.seed)