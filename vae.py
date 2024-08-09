import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, airfoil_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(airfoil_dim, 256)
        self.fc2 = nn.LeakyReLU(0.2, inplace=True)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.LeakyReLU(0.2, inplace=True)
        self.fc5_mu = nn.Linear(128, latent_dim)
        self.fc5_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        mu = self.fc5_mu(x)
        logvar = self.fc5_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, airfoil_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, airfoil_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class VAE(nn.Module):
    def __init__(self, airfoil_dim, latent_dim):
        super(VAE, self).__init__()
        self.enc = Encoder(airfoil_dim, latent_dim)
        self.dec = Decoder(latent_dim, airfoil_dim)

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)