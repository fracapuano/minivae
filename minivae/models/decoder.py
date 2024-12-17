import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Poisson, NegativeBinomial
from minivae.models.distributions import ZeroInflatedDistribution


class DecoderHead(nn.Module):
    def __init__(self, hidden_dims:list[int], output_dim:int):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build layers dynamically, based on hidden_dims
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        
        # leaves last layer hanging, for subclasses to overwrite
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")

class GaussianHead(DecoderHead):
    """
    Subclasses DecoderHead and implements the forward method for the normal distribution.
    The forward pass returns a Gaussian distribution with mean mu and log variance logvar.

    Mean and log variance are computed from a latent vector, fed through the decoder.
    """
    def __init__(self, hidden_dims, output_dim):
        super().__init__(hidden_dims, output_dim)
        self.mu = nn.Linear(hidden_dims[-1], output_dim)
        self.logvar = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x):
        # Feed the latent vector through the decoder
        x = self.layers(x)

        # Project intermediate output to mean and log variance
        mu, logvar = self.mu(x), self.logvar(x)

        # log(var) -> std=sqrt(e^(log(var)))=e^(1/2 * log(var))
        std = torch.exp(0.5 * logvar)

        return Normal(mu, std)

class PoissonHead(DecoderHead):
    """
    Subclasses DecoderHead and implements the forward method for the Poisson distribution.
    The forward pass returns a Poisson distribution with rates computed through the decoder.

    Rate is computed from a latent vector, fed through the decoder.
    """
    def __init__(self, hidden_dims, output_dim):
        super().__init__(hidden_dims, output_dim)
        self.rates = nn.Linear(self.hidden_dims[-1], self.output_dim)
        
    def forward(self, x):
        # Feed the latent vector through the decoder
        x = self.layers(x)

        # Project intermediate output to rate
        rate = F.relu(self.rates(x)) + 1e-6
        
        return Poisson(rate)

class NegativeBinomialHead(DecoderHead):
    """
    Subclasses DecoderHead and implements the forward method for the Negative Binomial distribution.
    The forward pass returns a Negative Binomial distribution with parameters mu and alpha.

    Parameters are computed from a latent vector, fed through the decoder.
    """
    def __init__(self, hidden_dims, output_dim):
        super().__init__(hidden_dims, output_dim)
        self.mu = nn.Linear(hidden_dims[-1], output_dim)
        self.alpha = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x):
        # Feed the latent vector through the decoder
        x = self.layers(x)

        # Project intermediate output to mu and alpha
        mu, alpha = self.mu(x), self.alpha(x)

        # Ensure mu and alpha are positive
        mu = F.softplus(mu)
        alpha = F.softplus(alpha)

        return NegativeBinomial(total_count=alpha, probs=mu/(mu + alpha))

class ZeroInflatedPoissonHead(DecoderHead):
    def __init__(self, hidden_dims, output_dim):
        super().__init__(hidden_dims, output_dim)
        self.base_distribution = PoissonHead(hidden_dims, output_dim)
        self.rho = nn.Parameter(torch.tensor([0.5]))
        
    def forward(self, x):
        vanilla_distr = self.base_distribution(x)
        return ZeroInflatedDistribution(vanilla_distr, self.rho)

class ZeroInflatedNegativeBinomialHead(DecoderHead):
    def __init__(self, hidden_dims, output_dim):
        super().__init__(hidden_dims, output_dim)
        self.base_distribution = NegativeBinomialHead(hidden_dims, output_dim)
        self.rho = nn.Parameter(torch.tensor([0.5]))
        
    def forward(self, x):
        vanilla_distr = self.base_distribution(x)
        return ZeroInflatedDistribution(vanilla_distr, self.rho)

decoder_heads = {
    "gaussian": GaussianHead,
    "poisson": PoissonHead,
    "negative_binomial": NegativeBinomialHead,
    "zip": ZeroInflatedPoissonHead,
    "zi_nb": ZeroInflatedNegativeBinomialHead
}
