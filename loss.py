import torch
from torch.distributions import Normal

class VAELoss:
    def __init__(self, kl_weight=1.0):
        """
        Args:
            kl_weight (float): Weight for KL divergence term.
        """
        self.kl_weight = kl_weight

    def __call__(self, outputs, x, warm_up_weight=1.0):
        """
        Compute VAE loss based on model outputs
        
        Args:
            outputs: Tuple of (reconstruction_dist, mu, logvar) for VAE
                    or (reconstruction_dist, mu, logvar, mixture) for GMVAE
            x (torch.Tensor): Input data [batch_size, input_dim]
            warm_up_weight (float): Weight for KL term warm-up
            
        Returns:
            tuple: (total_loss, dict of metrics)
        """
        reconstruction_dist, mu, logvar = outputs

        # 1. Empirical likelihood of x under the reconstruction distribution
        likelihood = reconstruction_dist.log_prob(x).sum(dim=1).mean()

        # KL divergence has closed-form expression versus standard normal prior
        kl_div = (0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1, dim=1)).mean()
        
        # ELBO, with warmup
        elbo = likelihood - (self.kl_weight * warm_up_weight * kl_div)
        # VAE are maximized using the surrogate ELBO-objective
        loss = -elbo

        info = {
            "elbo": elbo,
            "likelihood": likelihood,
            "kl_div": kl_div
        }

        return loss, info

class GMVAELoss:
    def __init__(self, kl_weight=1.0):
        """
        Args:
            kl_weight (float): Weight for KL divergence term.
        """
        self.kl_weight = kl_weight

    def __call__(self,
                 outputs,
                 x,
                 warm_up_weight=1.0
                ):
        """
        Compute GMVAE loss based on model outputs
        
        Args:
            outputs (tuple): (reconstruction_dist, mu, logvar, z, y_logits, y_onehot)
            x (torch.Tensor): Input data [batch_size, input_dim]
            warm_up_weight (float): Weight for KL term warm-up
            
        Returns:
            tuple: (total_loss, dict of metrics)
        """
        reconstruction_dist, mu, logvar, z, y_logits, y_onehot = outputs
        
        # 1. Reconstruction loss (negative log likelihood)
        log_likelihood = reconstruction_dist.log_prob(x).sum(dim=1).mean()
        
        # 2. KL divergence for z given y (using closed form for normal distributions)
        kl_z = 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1, dim=1).mean()
        
        # 3. KL divergence for categorical y (versus uniform prior)
        num_components = y_logits.size(-1)
        y_probs = torch.softmax(y_logits, dim=-1)
        # here I am assuming a uniform prior on distribution of Y
        uniform_prior = torch.ones_like(y_probs) / num_components
        kl_y = torch.sum(y_probs * (torch.log(y_probs + 1e-8) - torch.log(uniform_prior)), dim=1).mean()
        
        # Total KL divergence
        kl_div = kl_z + kl_y
        
        # ELBO with warmup
        elbo = log_likelihood - (self.kl_weight * warm_up_weight * kl_div)
        # GMVAE loss (negative ELBO)
        loss = -elbo
        
        info = {
            "elbo": elbo,
            "log_likelihood": log_likelihood,
            "kl_div": kl_div,
            "kl_z": kl_z,
            "kl_y": kl_y
        }
        
        return loss, info

# Example usage
if __name__ == "__main__":
    from gmvae import GMVAE
    # Dummy data
    batch_size, input_dim = 32, 10
    x = torch.randn(batch_size, input_dim)
    
    model = GMVAE(input_dim, 10, 10, reconstruction_dist="gaussian")
    outputs = model(x)  # This will return all necessary components
    
    gmvae_loss = GMVAELoss()
    
    # Test standard GMVAE loss
    loss, info = gmvae_loss(
        reconstruction_dist=outputs[0],
        mu=outputs[1],
        logvar=outputs[2],
        y_logits=outputs[4],
        x=x,
        warm_up_weight=1
    )
    
    print("GMVAE Loss:")
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {info}")