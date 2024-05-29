import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder
from Decoder import Decoder
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

class VAE(nn.Module):
    def __init__(self, encoder=None, decoder=None):
        super(VAE, self).__init__()
        
        if not encoder:
            self.encoder = Encoder()
        if not decoder:
            self.decoder = Decoder()
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD
    
    def train(self, dataloader, epochs=100, lr=2e-3, gamma=0.95):
        optimizer = Adam(self.parameters(), lr=lr)
        lr_s = StepLR(optimizer, step_size=10, gamma=gamma)
        for epoch in range(epochs):
            train_loss = 0
            
            for batch_idx, (data, _) in enumerate(dataloader):
                data = data.cuda()
                optimizer.zero_grad()
                
                recon_batch, mu, log_var = self(data)
                loss = self.loss_function(recon_batch, data, mu, log_var)
                
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            lr_s.step()
            print("Epoch: {}, Loss={:.4f}, LR={:.4f}".format(epoch, train_loss / len(dataloader.dataset), lr_s.get_last_lr()[0]))