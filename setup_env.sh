# sets up a functioning environment on a GPU deployment

curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install Miniconda silently to home directory
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3

# Initialize shell for conda
source ~/miniconda3/bin/activate

# Create virtual environment with Python 3.10
conda create -y -n scenv python=3.10

# Activate the environment
conda activate scenv

# Install poetry
pip install poetry

# Run poetry install for dependencies
poetry install