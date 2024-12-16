import torch
import torch.nn as nn
from decoder import decoder_heads


class VAE(nn.Module):
    def __init__(
            self, 
            input_dim, 
            latent_dim, 
            hidden_dims=[128, 64], 
            reconstruction_dist="poisson",
        ):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        if reconstruction_dist not in decoder_heads:
            raise ValueError(
                f"Unsupported distribution: {reconstruction_dist}."
                " Values supported: {decoder_heads.keys()}"
            )

        # creates a decoder, with an header determined by the reconstruction_dist
        self.decoder = decoder_heads[reconstruction_dist](
            [latent_dim] + list(reversed(hidden_dims)), input_dim
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.encoder_mu(h), self.encoder_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        # maps input datapoints to the corresponding distribution
        mu, logvar = self.encode(x)
        # samples from latent distribution with reparam. trick
        z = self.reparameterize(mu, logvar)
        # maps latent sample to recostrunction distribution
        reconstruction_dist = self.decode(z)
        
        return reconstruction_dist, mu, logvar


if __name__=="__main__":
    # dummy run
    x = torch.randn(
        (16, 32738)  # (B, S)
        )

    scVAE = VAE(
        32738,
        10, 
        reconstruction_dist="gaussian"
    )

    print(scVAE)

    distr, mu, logvar = scVAE(x)
    print(distr)
    print(mu.shape)
    print(logvar.shape)

    print(distr.log_prob(x).sum(dim=1).mean())
