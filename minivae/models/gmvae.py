import torch
import torch.nn as nn
import torch.nn.functional as F

from minivae.models.vae import VAE


class GMVAE(VAE):
    def __init__(self, 
                 input_dim, 
                 latent_dim,
                 n_components,
                 hidden_dims=[128, 64],
                 reconstruction_dist="poisson",

                ):
        # Initialize parent VAE class
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims, 
            reconstruction_dist=reconstruction_dist
        )
        # deleting VAE-like encoding structure
        del self.encoder, self.encoder_mu, self.encoder_logvar

        # number of components for the assumed MoG underlying z
        self.n_components = n_components
        
        # dividing the hidden dims into two chunks: P(Y|X), P(Z|X,Y)
        first_half, second_half = hidden_dims[:len(hidden_dims)//2], hidden_dims[len(hidden_dims)//2:]

        """P(y|x) is parametrized using a first encoder route."""
        layers = []
        prev_dim = input_dim
        for dim in first_half:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim

        layers.append(
            nn.Linear(first_half[-1], n_components)
        )
        # mapping an input data point to the logits corresponding to the latent variable, y
        self.x_to_latent_y = nn.Sequential(*layers)

        """P(z|x,y) is parametrized using a second encoder route"""
        layers = []
        prev_dim = input_dim + self.n_components
        for dim in second_half:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim

        self.x_y_to_latent_z = nn.Sequential(*layers)

        self.encoder_mu = nn.Linear(second_half[-1], latent_dim)
        self.encoder_logvar = nn.Linear(second_half[-1], latent_dim)
        
    def encode(self, x):
        y_logits = self.x_to_latent_y(x)
        y_onehot = F.gumbel_softmax(y_logits)
        
        h = self.x_y_to_latent_z(torch.hstack((x, y_onehot)))
        return self.encoder_mu(h), self.encoder_logvar(h), y_logits, y_onehot

    def forward(self, x):
        mu, logvar, y_logits, y_onehot = self.encode(x)
        # obtain z, conditioned on (X,Y)
        z = self.reparameterize(mu, logvar)
        
        # Decode the sample in latent space
        reconstruction_dist = self.decoder(z)
        
        return reconstruction_dist, mu, logvar, z, y_logits, y_onehot


if __name__=="__main__":
    # dummy run
    x = torch.randn(
        (16, 32738)  # (B, S)
        )

    scVAE = GMVAE(
        32738,
        10,
        9,
        reconstruction_dist="gaussian"
    )

    print(scVAE)

    distr, mu, logvar, z, y_logits, y_onehot = scVAE(x)
    print(distr)
    print(mu.shape)
    print(logvar.shape)

    print(distr.log_prob(x).sum(dim=1).mean())
