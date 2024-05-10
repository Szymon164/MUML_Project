import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size=784):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 2000)
        self.fc_mean = nn.Linear(2000, 10)
        self.fc_log_var = nn.Linear(2000, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc_mean(x), self.fc_log_var(x)
    
class Decoder(nn.Module):
    def __init__(self, output_size=784):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(10, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.sigmoid(self.fc4(x))

class VAE(nn.Module):
    def __init__(self, size=784, encoder=None, decoder=None):
        super(VAE, self).__init__()
        
        if not encoder:
            self.encoder = Encoder(size)
        if not decoder:
            self.decoder = Decoder(size)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var