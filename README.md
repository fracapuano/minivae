# minivae

A minimalist implementation of Variational Auto Encoders (VAEs) in pure PyTorch, with a focus on discrete/count data reconstruction. This implementation achieves the same functionality as scVAE, with a fraction of the code for improved accessibility (3k+ LOC vs. scVAE's 30k!). 

We open-source the data used here, to make it extra-accessible by the community. For enhanced reproducibility, we also freely distribute our training runs via WandB. 

- 🐝 Training Runs, [here](https://wandb.ai/francescocapuano/scRNA-VAE/overview)
- 🤗 Data, [fracapuano/scRNA](https://huggingface.co/fracapuano/datasets/scRNA).


## Key Features

- 🎯 Specialized for discrete/count data reconstruction, such as single-cell sequencing data
- 🧬 Tailored for genomics, and discrete RL/MDPs
- 💡 Pure PyTorch implementation - minimal dependencies and 2024 stack (TF1 🤮)!
- 🚀 10x code reduction, while maintaining full functionality (this is a repo that is meant to be easy to read!)

## Quick Start: Launch Training

```python
# change to gmvae to use gaussian mixture models
from vae import VAE

from torch.utils.data import DataLoader

# Initialize the VAE
model = VAE(
    input_dim=32738,
    latent_dim=100,
    hidden_dims=[256, 128, 64],
    reconstruction_dist="poisson"
)

# Create your dataloader
train_loader = DataLoader(your_dataset, batch_size=32)

# Train with configuration from yaml
from train import train_epoch
import yaml

with open("configs/train_vae.yaml") as f:
    config = yaml.safe_load(f)

# runs one epoch of training
train_epoch(model, train_loader, optimizer, loss_fn, device="cuda")
```

For the complete training scripts, check out `train.py`!


## Available Models

### (Vanilla) VAE

- Flexible encoder/decoder architectures

- Multiple reconstruction distributions (Gaussian, Poisson, Negative Binomial)

- KL annealing for better training stability

### Gaussian Mixture VAE (GMVAE)

- Extends standard VAE with mixture modeling over the latent variable, $z$


## Specify Training Configuration

Training parameters can be easily configured through YAML files:
```yaml
project: "scRNA-VAE"
architecture: "VAE"  # or "GMVAE"
dataset: "scRNA-2k"  # subsampled huggingface dataset
input_dim: 32738
latent_dim: 100
hidden_dims: [256, 128, 64]
learning_rate: 3.0e-4
batch_size: 32
epochs: 20
reconstruction_dist: "poisson"
```


## License

MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

This project was developed as part of the "Introduction to Probabilistic Graphical Methods" course at [MVA](https://master-mva.com) (ENS Paris Saclay), 2024.