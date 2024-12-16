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
        
# Example usage
if __name__ == "__main__":
    # Dummy data
    batch_size, input_dim = 32, 10
    x = torch.randn(batch_size, input_dim)
    
    vae_loss = VAELoss(kl_weight=1.0)
    
    reconstruction_dist = Normal(torch.zeros_like(x), torch.ones_like(x))
    mu = torch.zeros(batch_size, 2)
    logvar = torch.zeros(batch_size, 2)
    
    # Test standard VAE loss
    outputs = (reconstruction_dist, mu, logvar)
    loss, info = vae_loss(
        outputs=outputs,
        x=x,
        warm_up_weight=1
    )
    
    print("VAE Loss:")
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {info}")