import torch
import torch.distributions as D


class ZeroInflatedDistribution(torch.distributions.Distribution):
    """
    Defines a wrapper around Pytorch distributions for zero-inflation,
    as per Grønbech et al., 2020.
    """
    def __init__(self, dist, zero_inflation):
        super().__init__(validate_args=False)
        self.base_dist = dist
        self.rho = zero_inflation
        
    def log_prob(self, value):
        base_log_prob = self.base_dist.log_prob(value)
        is_zero = (value <= 1e-9)
        
        # P(x) = rho + (1-rho) * P(x), when x = 0
        log_prob_zero = torch.log(
            self.rho + (1 - self.rho) * torch.exp(base_log_prob)
        )
        
        # P(x) = (1-rho) * P(x), when x > 0
        log_prob_non_zero = torch.log(1 - self.rho) + base_log_prob
        
        return torch.where(is_zero, log_prob_zero, log_prob_non_zero)
    
    def mean(self):
        return (1 - self.zero_inflation) * self.base_dist.mean

def zero_inflated_poisson(rate, zero_inflation):
    """Creates a zero-inflated Poisson distribution"""
    return ZeroInflatedDistribution(
        dist=D.Poisson(rate),
        zero_inflation=zero_inflation
    )

def zero_inflated_negative_binomial(total_count, probs, zero_inflation):
    """Creates a zero-inflated Negative Binomial distribution"""
    return ZeroInflatedDistribution(
        dist=D.NegativeBinomial(total_count, probs),
        zero_inflation=zero_inflation
    )

