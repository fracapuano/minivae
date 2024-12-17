import wandb
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import yaml
import os

from vae import VAE
from gmvae import GMVAE

from loss import VAELoss, GMVAELoss

def collate_fn(batch):
    # Extract sequences and convert to tensor
    sequences = [
        torch.tensor(item['sequence'], dtype=torch.float32) for item in batch
    ]
    return torch.stack(sequences)

def create_dataloader(split, batch_size=32):
    """Creates a DataLoader for a specific split of the scRNA dataset"""
    dataset = load_dataset(
        "fracapuano/scRNA-2k", split=split, streaming=True).\
        rename_column("gene_expression", "sequence")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=1
    )

def train_epoch(model, train_loader, optimizer, loss_fn, device, warm_up_weight=1.0):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        x = batch.to(device)
        
        # Forward pass
        outputs = model(x)
        
        # Compute loss
        loss, metrics = loss_fn(
            outputs=outputs,
            x=x,
            warm_up_weight=warm_up_weight
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        kl_mixtures = {
            "train/kl_z": metrics.get("kl_z", -1),
            "train/kl_y": metrics.get("kl_y", -1)
        }
        kl_mixtures = {
            k: v.item() if isinstance(v, torch.Tensor) else v for k,v in kl_mixtures.items()
        }
        
        # Log metrics
        wandb.log({
            "train/loss": loss.item(),
            "train/elbo": metrics["elbo"].item(),
            "train/log_likelihood": metrics["log_likelihood"].item(),
            "train/kl_div": metrics["kl_div"].item(),
        } | kl_mixtures
        )
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            x = batch.to(device)
            
            # Forward pass
            outputs = model(x)
            
            # Compute loss
            loss, metrics = loss_fn(
                outputs=outputs,
                x=x
            )

            kl_mixtures = {
            "train/kl_z": metrics.get("kl_z", -1),
            "train/kl_y": metrics.get("kl_y", -1)
            }
            kl_mixtures = {
                k: v.item() if isinstance(v, torch.Tensor) else v for k,v in kl_mixtures.items()
            }
            
            # Log metrics
            wandb.log({
                "val/loss": loss.item(),
                "val/elbo": metrics["elbo"].item(),
                "val/log_likelihood": metrics["log_likelihood"].item(),
                "val/kl_div": metrics["kl_div"].item()
            } | kl_mixtures
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_model(config):
    """
    Factory function to create model instances based on config.
    
    Args:
        config (dict): Configuration dictionary containing model parameters
            Required fields:
            - architecture: "VAE" or "GMVAE"
            - input_dim: Input dimension
            - latent_dim: Latent space dimension
            - hidden_dims: List of hidden layer dimensions
            - reconstruction_dist: Type of reconstruction distribution
            For GMVAE only:
            - nclusters: Number of Gaussian components
            
    Returns:
        torch.nn.Module: Instantiated model (VAE or GMVAE)
    """
    architecture = config["architecture"].upper()
    
    if architecture == "VAE":
        model = VAE(
            input_dim=config["input_dim"],
            latent_dim=config["latent_dim"],
            hidden_dims=config["hidden_dims"],
            reconstruction_dist=config["reconstruction_dist"]
        )
    elif architecture == "GMVAE":
        model = GMVAE(
            input_dim=config["input_dim"],
            latent_dim=config["latent_dim"],
            n_components=config["n_components"],
            hidden_dims=config["hidden_dims"],
            reconstruction_dist=config["reconstruction_dist"]
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose 'VAE' or 'GMVAE'")
    
    return model

def get_loss(config):
    """Routes the current configuration dictionary chosen to a given loss function, specific for a problem"""
    architecture = config["architecture"].upper()
    
    if architecture == "VAE":
        return VAELoss()
    
    elif architecture == "GMVAE":
        return GMVAELoss()

    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose 'VAE' or 'GMVAE'")

def main():
    # Load configuration
    config = load_config('config.yaml')
    
    # Initialize wandb with loaded config
    run = wandb.init(
        project=config['project'],
        config=config
    )
    
    # Create a unique model path using the wandb run ID
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"best_model_{run.id}.pt")

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )

    model = get_model(config).to(device)

    # Create data loaders
    train_loader = create_dataloader("train", batch_size=config['batch_size'])
    val_loader = create_dataloader("validation", batch_size=config['batch_size'])
    
    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate']
    )
    
    loss_fn = get_loss(config)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        # Compute warm-up weight (increasing linear schedule over 10 epochs)
        warm_up_weight = min(1.0, (epoch + 1) / 10)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, 
            device, warm_up_weight
        )
        
        # Validate
        val_loss = validate(model, val_loader, loss_fn, device)
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch,
            "train/epoch_loss": train_loss,
            "val/epoch_loss": val_loss,
            "warm_up_weight": warm_up_weight
        })
        
        # Save best model with unique identifier
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)
        
        print(
            f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )
    
    wandb.finish()

if __name__ == "__main__":
    main()

