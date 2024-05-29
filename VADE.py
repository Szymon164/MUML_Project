import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from sklearn.mixture import GaussianMixture
from Encoder import Encoder
from Decoder import Decoder

class VaDE(nn.Module):
    def __init__(self, encoder=None, decoder=None):
        super(VaDE, self).__init__()
        self.n_clusters = 10
        self.cuda = torch.cuda.is_available()
        self.encoder = Encoder() if not encoder else encoder
        self.decoder = Decoder() if not decoder else decoder
        
        self.pi_ = nn.Parameter(
            torch.FloatTensor(
                self.n_clusters,
            ).fill_(1)
            / self.n_clusters,
            requires_grad=True,
        )

        self.mu_c = nn.Parameter(
            torch.FloatTensor(self.n_clusters, 10),
            requires_grad=True,
        )
        self.log_sigma2_c = nn.Parameter(
            torch.FloatTensor(self.n_clusters, 10),
            requires_grad=True,
        )

    def init_gmm(self, dataloader):
        Z = []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.cuda()
                z1, _ = self.encoder(x)
                Z.append(z1)
        Z = torch.cat(Z, 0).detach().cpu().numpy()
        gmm = GaussianMixture(n_components=self.n_clusters, covariance_type="diag")
        gmm.fit(Z)
        if self.cuda:
            self.pi_.data = torch.from_numpy(gmm.weights_).cuda().float()
            self.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
            self.log_sigma2_c.data = torch.log(
                torch.from_numpy(gmm.covariances_).cuda().float()
            )
        else:
            self.pi_.data = torch.from_numpy(gmm.weights_).float()
            self.mu_c.data = torch.from_numpy(gmm.means_).float()
            self.log_sigma2_c.data = torch.log(
                torch.from_numpy(gmm.covariances_).float()
            )

    def train(self, dataloader, epochs=100, lr=2e-3, gamma=0.95):
        self.init_gmm(dataloader)
        opti = Adam(self.parameters(), lr=lr)
        lr_s = StepLR(opti, step_size=10, gamma=gamma)
        writer = SummaryWriter("./logs")
        epoch_bar = range(epochs)
        for epoch in epoch_bar:
            L = 0
            for x, _ in dataloader:
                x = x.cuda()
                loss = self.ELBO_Loss(x)
                opti.zero_grad()
                loss.backward()
                opti.step()
                L += loss.item()
            lr_s.step()
            writer.add_scalar("loss", L / len(dataloader), epoch)
            writer.add_scalar("lr", lr_s.get_last_lr()[0], epoch)
            print(
                "Epoch: {}, Loss={:.4f}, LR={:.4f}".format(epoch, L / len(dataloader), lr_s.get_last_lr()[0])
            )

    def predict_proba(self, x):
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        y_c = torch.exp(
            torch.log(self.pi_.unsqueeze(0))
            + self.gaussian_pdfs_log(z)
        )
        return y_c


    def predict(self, x):
        y_c = self.predict_proba(x)
        y = y_c.detach().cpu().numpy()
        return np.argmax(y, axis=1)

    def ELBO_Loss(self, x, det=1e-10):
        L_rec = 0
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        x_pro = self.decoder(z)  # x_pro sometimes has nans
        try:
            L_rec += F.binary_cross_entropy(x_pro, x)
        except:
            print(x_pro.min(), x_pro.max())
        L_rec = L_rec
        Loss = L_rec * x.size(1)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        y_c = (
            torch.exp(
                torch.log(self.pi_.unsqueeze(0))
                + self.gaussian_pdfs_log(z)
            )
            + det
        )
        y_c = y_c / (y_c.sum(1).view(-1, 1))  # batch_size*Clusters
        Loss += 0.5 * torch.mean(
            torch.sum(
                y_c
                * torch.sum(
                    self.log_sigma2_c.unsqueeze(0)
                    + torch.exp(
                        z_sigma2_log.unsqueeze(1) - self.log_sigma2_c.unsqueeze(0)
                    )
                    + (z_mu.unsqueeze(1) - self.mu_c.unsqueeze(0)).pow(2)
                    / torch.exp(self.log_sigma2_c.unsqueeze(0)),
                    2,
                ),
                1,
            )
        )
        Loss -= torch.mean(
            torch.sum(y_c * torch.log(self.pi_.unsqueeze(0) / (y_c)), 1)
        ) + 0.5 * torch.mean(torch.sum(1 + z_sigma2_log, 1))
        return Loss

    def gaussian_pdfs_log(self, x):
        G = []
        for c in range(self.n_clusters):
            G.append(
                self.gaussian_pdf_log(
                    x, self.mu_c[c : c + 1, :], self.log_sigma2_c[c : c + 1, :]
                ).view(-1, 1)
            )
        return torch.cat(G, 1)

    def sample_clusters(self):
        _, n_features = self.mu_c.shape
        rng = np.random.default_rng()
        n_samples_comp = np.ones(10, dtype=np.int8)
        means = self.mu_c.detach().cpu().numpy()
        covs = torch.exp(self.log_sigma2_c).detach().cpu().numpy()
        if True:
            X = np.vstack(
                [
                    mean
                    + rng.standard_normal(size=(sample, n_features))
                    * np.sqrt(covariance)
                    for (mean, covariance, sample) in zip(
                        means, covs, n_samples_comp
                    )
                ]
            )

        y = np.concatenate(
            [np.full(sample, j, dtype=int) for j, sample in enumerate(n_samples_comp)]
        )
        return X,y

    def gaussian_pdf_log(self, x, mu, log_sigma2):
        return -0.5 * (
            torch.sum(
                np.log(np.pi * 2)
                + log_sigma2
                + (x - mu).pow(2) / torch.exp(log_sigma2),
                1,
            )
        )
